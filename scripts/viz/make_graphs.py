import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from multi_robot_svbp.costs.trajectory_costs import RunningDeltaCost, TerminalDeltaCost


colours = ["tab:orange",
           "tab:pink",
           "tab:cyan",
           "tab:green",
           "tab:purple",
           "tab:red",
           "tab:blue",
           "tab:olive",
           "tab:brown",
           "tab:grey"]
# ROOT_PATH = "output/exp"
ROOT_PATH = "/hdd/jana/stein_bp/planar-results-sept13-final-maybe"
STATE_FILE = "states.npy"

plt.rcParams.update({'font.size': 28, 'font.family': 'serif'})


def make_costs(c_pos=0., c_vel=0.25, c_u=0.2, c_term=6.,
               dim=2, horizon=1, goal=None,
               tensor_kwargs={"device": "cpu", "dtype": torch.float}):
    """Utility function for creating the costs for the point swarm problem."""
    goal = torch.as_tensor(goal).to(**tensor_kwargs)
    running_cost_Qs = (c_pos * torch.eye(dim), c_vel * torch.eye(dim), c_u * torch.eye(dim))
    running_cost_x_bars = (goal, torch.zeros_like(goal), torch.zeros_like(goal))
    running_cost_fn = RunningDeltaCost(Qs=running_cost_Qs, x_bars=running_cost_x_bars,
                                       tensor_kwargs=tensor_kwargs)
    terminal_cost_Qs = (c_term * torch.eye(dim), 0 * torch.eye(dim), 0 * torch.eye(dim))
    terminal_cost_x_bars = (goal, torch.zeros_like(goal), torch.zeros_like(goal))
    terminal_cost_fn = TerminalDeltaCost(Qs=terminal_cost_Qs, x_bars=terminal_cost_x_bars,
                                         tensor_kwargs=tensor_kwargs)

    return running_cost_fn, terminal_cost_fn


def euclidean_path_length(path):
    """Calculates the length along a path. Path is a tensor with dimension (T, D)."""
    diff = path[:-1, :] - path[1:, :]
    dist = (diff ** 2).sum(-1)
    return np.sqrt(dist).sum()


def euclidean_distance(x, y, squared=True):
    """Returns squared Euclidean distance pairwise between x and y.

    If x is size (N, D) and y is size (M, D), returns a matrix of size (N, M)
    where element (i, j) is the distance between x[i, :] and y[j, :]. If x is
    size (N, D) and y is size (D,), returns a vector of length (N,) where each
    element is the distance from a row in x to y. If both have size (D,)
    returns a number.
    """
    if x.ndim == 1 and y.ndim == 1:
        # Shape is (D,), (D,).
        diffs = x - y
    else:
        diffs = (np.expand_dims(x, 1) - y).squeeze()
    dist = np.sum(diffs**2, axis=-1)
    if not squared:
        dist = np.sqrt(dist)
    return dist


def terminal_point(traj, goal, thresh=0.30):
    dists = euclidean_distance(traj, goal, squared=False)
    idx, = np.nonzero(dists < thresh)
    if idx.shape[0] == 0:
        return -1
    return idx[0]


def detect_collision(states, radius=0.2, tol=0.001):
    T, N, D = states.shape
    collisions = []
    for t in range(T):
        dists = euclidean_distance(states[t, :, :2], states[t, :, :2], squared=False)
        coll = dists <= 2 * radius - tol
        np.fill_diagonal(coll, False)
        coll_per_robot = coll.any(0)
        collisions.append(coll_per_robot)

    collisions = np.array(collisions)

    return collisions.any(0)  # length N boolean array, where i-th index is true if robot i collided.


def get_goals(scenes):
    goals = {}
    for scene in scenes:
        with open(f"data/scenes/{scene}.yml", 'r') as f:
            scene_data = yaml.load(f, Loader=yaml.Loader)

        goals[scene] = np.array(scene_data["goals"])

    return goals


def get_starts(scenes):
    starts = {}
    for scene in scenes:
        with open(f"data/scenes/{scene}.yml", 'r') as f:
            scene_data = yaml.load(f, Loader=yaml.Loader)

        starts[scene] = np.array(scene_data["starts"])

    return starts


def get_all_dists(scenes, method, runs, goals, dt=0.1):
    path = os.path.join(ROOT_PATH, method)
    dists = {}
    times = {}
    for scene in scenes:
        dists[scene] = []
        times[scene] = []
        for run in range(runs):
            run_path = os.path.join(path, scene, str(run), STATE_FILE)
            if not os.path.exists(run_path):
                print("Missing!!", run_path)
                continue
            states = np.load(run_path)

            T, N, D = states.shape
            for n in range(N):
                term_idx = terminal_point(states[:, n, :2], goals[scene][n])
                if term_idx < 0:
                    print("Failed:", n, run_path)
                    continue
                d = euclidean_path_length(states[:(term_idx + 1), n, :2])
                dists[scene].append(d)
                times[scene].append(term_idx)

        dists[scene] = np.array(dists[scene])
        times[scene] = dt * np.array(times[scene])

    return dists, times


def get_all_costs(scenes, method, runs, goals, dt=0.1, tensor_kwargs={"device": "cpu", "dtype": torch.float}):
    path = os.path.join(ROOT_PATH, method)
    costs = {}
    for scene in scenes:
        costs[scene] = []
        for run in range(runs):
            states_path = os.path.join(path, scene, str(run), STATE_FILE)
            states = np.load(states_path)
            T, N, D = states.shape

            if method.startswith("sbp"):
                ctrl_path = os.path.join(path, scene, str(run), "ctrl.npy")
                ctrls = np.load(ctrl_path)
            else:
                ctrls = [np.load(os.path.join(path, scene, str(run), f"means_{i:04d}.npy"))[:, 0, :]
                         for i in range(T - 1)]
                ctrls = np.stack(ctrls)

            traj = np.concatenate([states[1:, :, :], ctrls], axis=-1)

            for n in range(N):
                running_cost, term_cost = make_costs(c_pos=0.1, c_vel=0.35, c_u=0.5, c_term=6,
                                                     horizon=T, goal=goals[scene][n], tensor_kwargs=tensor_kwargs)
                traj_tensor = torch.as_tensor(traj[:, n, :], **tensor_kwargs)
                cost = running_cost(traj_tensor) + term_cost(traj_tensor)
                costs[scene].append(cost.item())

        costs[scene] = np.array(costs[scene])

    return costs


def get_all_success(scenes, method, runs, goals):
    path = os.path.join(ROOT_PATH, method)
    success = {}
    collisions = {}
    dists = {}
    for scene in scenes:
        success[scene] = []
        collisions[scene] = []
        dists[scene] = []
        for run in range(runs):
            run_path = os.path.join(path, scene, str(run), STATE_FILE)
            if not os.path.exists(run_path):
                print("Missing!!", run_path)
                continue
            states = np.load(run_path)
            collisions[scene].append(detect_collision(states))

            T, N, D = states.shape
            for n in range(N):
                term_idx = terminal_point(states[:, n, :2], goals[scene][n])
                success[scene].append(term_idx >= 0)
                dists[scene].append(euclidean_distance(states[-1, n, :2], goals[scene][n], squared=False))

        success[scene] = np.array(success[scene], dtype=bool)
        collisions[scene] = np.array(collisions[scene], dtype=bool).flatten()
        dists[scene] = np.array(dists[scene])

    return success, collisions, dists


# Get all the path lengths for each scene.
SCENES = [
    "swap_rooms",
    "squares_swap",
    "squares_3x3_cross",
    "obstacles_mix_circle",
    "end_of_lessons"
]
GOALS = get_goals(SCENES)
STARTS = get_starts(SCENES)
RUNS = 10

# PATH LENGTH METRICS.

METHODS = {"sbp": {"name": "SVBP\n(Ours)", "c": "tab:green"},
           "orca_2": {"name": "ORCA\n(20 cm)", "c": "tab:red"},
           "orca_4": {"name": "ORCA\n(40 cm)", "c": "tab:blue"},
           "gabp": {"name": "GaBP", "c": "tab:orange"}}
# ALL_METHODS = METHODS.keys()
ALL_METHODS = ["sbp", "gabp", "orca_2", "orca_4"]

all_dists = {}
all_times = {}
for m in ALL_METHODS:
    dists, times = get_all_dists(SCENES, m, RUNS, GOALS)
    all_dists.update({m: dists})
    all_times.update({m: times})

# Concatenate distances for all the scenes.
plot_dists = [np.concatenate([all_dists[m][scene] for scene in SCENES]) for m in ALL_METHODS]
means = [np.mean(dists) for dists in plot_dists]
max_pt = 0

colours = [METHODS[m]["c"] for m in ALL_METHODS]
alpha_colors = [cm.to_rgba(c, alpha=0.3) for c in colours]
names = [METHODS[m]["name"] for m in ALL_METHODS]

# PATH LENGTHS
plt.figure(dpi=300, figsize=(12, 6))

for i, dists in enumerate(plot_dists):
    if dists.shape[0] == 0:
        continue
    plt.scatter(i * np.ones(dists.shape[0], dtype=int), dists, c=colours[i])
    max_pt = max(max_pt, np.max(dists))
plt.bar(names, means, color=alpha_colors, edgecolor=colours)
plt.ylabel("Path length (meters)")
# plt.xticks(rotation=30, ha='right')
plt.ylim(top=max_pt + 0.1)  # For some reason, the bar chart does not automatically find the top point.
plt.tight_layout()

# PATH TIMES
plt.figure(dpi=300, figsize=(12, 6))

plot_times = [np.concatenate([all_times[m][scene] for scene in SCENES]) for m in ALL_METHODS]
means = [np.mean(times) for times in plot_times]
max_pt = 0

for i, times in enumerate(plot_times):
    if times.shape[0] == 0:
        continue
    plt.scatter(i * np.ones(times.shape[0], dtype=int), times, c=colours[i])
    max_pt = max(max_pt, np.max(times))
plt.bar(names, means, color=alpha_colors, edgecolor=colours)
plt.ylabel("Path time (seconds)")
# plt.xticks(rotation=30, ha='right')
plt.ylim(top=max_pt + 0.1)  # For some reason, the bar chart does not automatically find the top point.
plt.tight_layout()

# SUCCESS RATES

all_success = {}
all_collisions = {}
all_dists = {}
for m in ALL_METHODS:
    success, collisions, dists = get_all_success(SCENES, m, RUNS, GOALS)
    all_success.update({m: success})
    all_collisions.update({m: collisions})
    all_dists.update({m: dists})

    # Print collitions cases.
    for s in SCENES:
        try:
            coll = collisions[s].reshape((RUNS, -1))
        except:
            print("Skipping", m, s)
            continue
        for run in range(RUNS):
            if np.any(coll[run, :]):
                print("Collisions:", m, s, f"run {run}:", coll[run, :].nonzero())

# SUCCESS RATE
plot_success = [np.concatenate([all_success[m][scene] for scene in SCENES]) for m in ALL_METHODS]
plot_collisions = [np.concatenate([all_collisions[m][scene] for scene in SCENES]) for m in ALL_METHODS]
for m, succ, coll in zip(ALL_METHODS, plot_success, plot_collisions):
    goal_reached = np.nonzero(succ)[0].shape[0] / succ.shape[0]
    collisions = np.nonzero(coll)[0].shape[0] / coll.shape[0]
    print("GOAL REACHED", m, np.nonzero(succ)[0].shape[0], "/", succ.shape[0], goal_reached)
    print("COLLISIONS", m, np.nonzero(coll)[0].shape[0], "/", coll.shape[0], collisions)
plot_success = [100 * np.nonzero(succ * ~coll)[0].shape[0] / succ.shape[0]
                for succ, coll in zip(plot_success, plot_collisions)]
print(ALL_METHODS, plot_success)

plt.figure(dpi=300, figsize=(12, 6))
plt.grid(zorder=0)
plt.bar(names, plot_success, color=colours, zorder=3)
plt.ylabel("Success rate (%)")
plt.tight_layout()

# PASS RATE
MAX_DIST = 1.5
LINEWIDTH = 3
x = np.linspace(0, MAX_DIST)
plot_dists = [np.concatenate([all_dists[m][scene] for scene in SCENES]) for m in ALL_METHODS]

plt.figure(dpi=300, figsize=(12, 6))
plt.grid()

for m, dists, coll in zip(ALL_METHODS, plot_dists, plot_collisions):
    n_dists, = dists.shape
    success = ~coll * (dists < np.expand_dims(x, -1))
    pass_rate = np.count_nonzero(success, -1) / n_dists

    plt.plot(x, pass_rate * 100, label=METHODS[m]["name"].replace("\n", " "), linewidth=LINEWIDTH,
             c=METHODS[m]["c"])

plt.ylabel("Pass Rate (%)")
plt.xlabel("Error Threshold (m)")
plt.ylim(0)
plt.xlim(0)
plt.legend()
plt.tight_layout()

# Costs
all_costs = {}
for m in ["sbp", "gabp"]:
    costs = get_all_costs(SCENES, m, RUNS, GOALS)
    all_costs.update({m: costs})

plot_costs = [np.concatenate([all_costs[m][scene] for scene in SCENES]) for m in ["sbp", "gabp"]]
means = [np.mean(costs) for costs in plot_costs]
colours = [METHODS[m]["c"] for m in ["sbp", "gabp"]]
alpha_colors = [cm.to_rgba(c, alpha=0.3) for c in colours]
names = [METHODS[m]["name"] for m in ["sbp", "gabp"]]

plt.figure(dpi=300, figsize=(6, 6))

for i, costs in enumerate(plot_costs):
    if costs.shape[0] == 0:
        continue
    plt.scatter(i * np.ones(costs.shape[0], dtype=int), costs, c=colours[i])
    max_pt = max(max_pt, np.max(costs))

plt.grid(zorder=0)
plt.bar(names, means, color=alpha_colors, edgecolor=colours, zorder=3)
plt.ylabel("Cost")
plt.ylim(top=max_pt + 0.1)
plt.tight_layout()

plt.show()
