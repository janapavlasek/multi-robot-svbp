import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm

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
# ROOT_PATH = "/hdd/jana/stein_bp/mbot_results_sept24/logs/"
ROOT_PATH = "output"
RUNS = 5
BOTS = ["mbot_progress_01", "mbot_progress_02", "mbot_progress_03"]
LOG_BOT = "mbot_progress_03"
EXPS = ["no_obs_swap", "obs_swap"]
METHODS = ["svbp", "orca"]
METHOD_DATA = {
    "svbp": {"name": "SVBP (Ours)", "color": "tab:blue"},
    "orca": {"name": "ORCA", "color": "tab:orange"}
}
EXP_DATA = {"no_obs_swap": "No Obstacles", "obs_swap": "Obstacles"}

plt.rcParams.update({'font.size': 15, 'font.family': 'serif'})


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


def get_start_idx(poses, tol=0.02):
    start_pose = poses[0, :2]
    dists = euclidean_distance(poses[:, :2], start_pose, squared=False)
    # Where the robot has moved from the start position.
    close_idx, = np.nonzero(dists > tol)
    # Take the first index where the robot is detected as moved for at least 2 timesteps.
    close_idx_filtered, = np.nonzero((close_idx[1:] - close_idx[:-1]) == 1)
    close_idx = close_idx[close_idx_filtered[0]]
    return close_idx - 1


def get_end_idx(poses, tol=0.02, pad=5):
    n_poses = poses.shape[0]
    end_pose = poses[-1, :2]
    dists = euclidean_distance(poses[:, :2], end_pose, squared=False)
    # Where the robot had not reached the end position.
    close_idx, = np.nonzero(dists < tol)
    # Take the first index where the robot is detected as moved for at least 2 timesteps.
    close_idx_filtered, = np.nonzero((close_idx[1:] - close_idx[:-1]) == 1)
    close_idx = close_idx[close_idx_filtered[0]]
    return np.minimum(close_idx + pad, n_poses - 1)


def get_run_start_time(bots, path):
    move_times = []
    stop_times = []
    for bot in bots:
        poses = np.load(os.path.join(path, bot + "_slam_pose.npy"))
        times = np.load(os.path.join(path, bot + "_slam_pose_times.npy"))

        start_idx = get_start_idx(poses)
        move_times.append(times[start_idx])

        # Get last pose in path.
        end_idx = get_end_idx(poses)
        stop_times.append(times[end_idx])

    return np.min(move_times), np.max(stop_times)


def run_goal_dists(poses, times, goal, start_time=None, end_time=None):
    if start_time is None:
        times = times - times[0]

    if start_time is not None:
        times = times - start_time
        valid_times, = np.nonzero(times >= 0)
        start_time_idx = valid_times[0]

        poses = poses[start_time_idx:, :2]
        times = times[start_time_idx:]

    if end_time is not None:
        valid_times, = np.nonzero(times <= end_time - start_time)
        poses = poses[valid_times, :]
        times = times[valid_times]

    goal_dists = euclidean_distance(poses, goal, squared=False)
    return goal_dists, times / 1e6


# TEST
dists = {exp: {m: {bot: {"data": [], "utimes": []} for bot in BOTS} for m in METHODS} for exp in EXPS}
term_dists = {m: {exp: [] for exp in EXPS} for m in METHODS}

for exp in EXPS:
    run_lengths = []
    for m in METHODS:
        for run in range(1, RUNS + 1):
            run_path = os.path.join(ROOT_PATH, m, exp, str(run))
            start_time, end_time = get_run_start_time(BOTS, run_path)
            run_lengths.append(end_time - start_time)
    run_length = np.max(run_lengths)

    for m in METHODS:
        for run in range(1, RUNS + 1):
            run_path = os.path.join(ROOT_PATH, m, exp, str(run))
            start_time, end_time = get_run_start_time(BOTS, run_path)
            end_time = start_time + run_length

            with open(os.path.join(run_path, "run_data.json")) as f:
                run_data = json.load(f)

            log_bot = run_data["name"].replace("-", "_")
            log_bot_start = np.array(run_data["start"])
            goals = {k.replace("-", "_"): np.array(v) for k, v in run_data["goals"].items()}

            for i, bot in enumerate(BOTS):
                poses = np.load(os.path.join(run_path, bot + "_slam_pose.npy"))
                times = np.load(os.path.join(run_path, bot + "_slam_pose_times.npy"))

                goal_dists, times = run_goal_dists(poses, times, goals[bot], start_time, end_time)
                print(exp, m, bot, type(times), times.shape)

                dists[exp][m][bot]["data"].append(goal_dists)
                dists[exp][m][bot]["utimes"].append(times)

                term_dists[m][exp].append(goal_dists[-1])

fig, exp_axs = plt.subplots(len(BOTS), len(EXPS), figsize=(8, 6))
for exp, axs in zip(EXPS, exp_axs.T):
    # fig, axs = plt.subplots(len(BOTS), figsize=(5, 5))
    for m in METHODS:
        colour = "tab:blue" if m == "svbp" else "tab:orange"
        for i, bot in enumerate(BOTS):
            # dists[exp][m][bot]["data"] = np.array(dists[exp][m][bot]["data"])
            # dists[exp][m][bot]["utimes"] = np.array(dists[exp][m][bot]["utimes"])
            # print(exp, m, bot, dists[exp][m][bot]["data"].shape, dists[exp][m][bot]["utimes"].shape)
            for run in range(RUNS):
                label = METHOD_DATA[m]["name"] if run == 1 and i == 0 and exp == EXPS[-1] else None
                goal_dists = dists[exp][m][bot]["data"][run]
                times = dists[exp][m][bot]["utimes"][run]
                axs[i].plot(times, goal_dists, label=label, c=METHOD_DATA[m]["color"], alpha=0.6)
            # axs[i].plot(times, goal_dists, label=label, c=colour, alpha=0.6)

    for i, ax in enumerate(axs):
        ax.grid()
        if i == 0 and exp == EXPS[-1]:
            ax.legend()
        if i == 0:
            ax.set_title(EXP_DATA[exp])
        ax.set_xlabel("Time (s)")
        if i == len(axs) // 2:
            ax.set_ylabel("Distance to goal (m)")
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.label_outer()

# Label robots on the right.
for i, ax in enumerate(exp_axs[:, -1]):
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(f"Robot #{i + 1}", rotation=-90, labelpad=20)

plt.tight_layout()

plt.figure()

MAX_DIST = 0.25
x = np.linspace(0, MAX_DIST)
thresh = np.expand_dims(x, -1)
for m in METHODS:
    dists_to_show = np.array([term_dists[m][exp] for exp in EXPS]).flatten()
    n_dists, = dists_to_show.shape
    pass_rate = np.count_nonzero(dists_to_show < thresh, -1) / n_dists

    plt.plot(x, pass_rate, label=m)
plt.legend()

plt.show()
