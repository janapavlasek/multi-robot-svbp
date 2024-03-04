import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from multi_robot_svbp.sim.map import DiffMap


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


def plot_scene(scene, starts=None, goals=None):
    with open(f"data/scenes/{scene}.yml", 'r') as f:
        scene_data = yaml.load(f, Loader=yaml.Loader)

    dmap = DiffMap(scene_data["map"])
    map_img = dmap.compute_binary_img()

    plt.figure(figsize=(8, 8))
    plt.imshow(map_img, extent=dmap.lims, cmap="Greys")

    if goals is not None:
        for i in range(goals.shape[0]):
            plt.scatter(goals[i, 0], goals[i, 1], c=colours[i % len(colours)],
                        marker='*', edgecolors="white", linewidths=2, s=900, zorder=3)

    shift = 0  # 0.1
    if starts is not None:
        for i in range(starts.shape[0]):
            plt.scatter(starts[i, 0], starts[i, 1] + shift, c=colours[i % len(colours)],
                        marker='o', edgecolors="white", s=1800, zorder=2)

    if starts is not None and goals is not None:
        shift = scene != "squares_3x3_cross"
        shift_size = 0.08
        for i in range(starts.shape[0]):
            x = np.array([starts[i, 0], goals[i, 0]])
            y = np.array([starts[i, 1], goals[i, 1]])
            if i < starts.shape[0] / 2:
                if scene == "squares_swap":
                    x = x - shift_size
                elif scene == "swap_rooms":
                    y = y - shift_size
            if shift and i >= starts.shape[0] / 2:
                if scene == "squares_swap":
                    x = x + shift_size
                elif scene == "swap_rooms":
                    y = y + shift_size
            plt.plot(x, y, c=colours[i % len(colours)], linestyle="--", linewidth=3, zorder=1)


def plot_scene_traj(method, scene, it, goals=None):
    with open(f"data/scenes/{scene}.yml", 'r') as f:
        scene_data = yaml.load(f, Loader=yaml.Loader)

    all_traj = np.load(os.path.join(ROOT_PATH, method, scene, str(it), STATE_FILE))

    T, N, D = all_traj.shape

    dmap = DiffMap(scene_data["map"])
    map_img = dmap.compute_binary_img()

    plt.figure(figsize=(8, 8))
    plt.imshow(map_img, extent=dmap.lims, cmap="Greys")
    for i in range(N):
        plt.plot(all_traj[:, i, 0], all_traj[:, i, 1], c=colours[i % len(colours)], linewidth=5)
        # t = np.arange(T - shift) / (T - shift)
        # plt.scatter(all_traj[:-shift, i, 0], all_traj[:-shift, i, 1], c=t, s=100)

        if goals is not None:
            plt.scatter(goals[i, 0], goals[i, 1], c=colours[i % len(colours)],
                        marker='X', edgecolors="white", s=500, zorder=3)


SCENES = [
    "swap_rooms",
    "squares_swap",
    "squares_3x3_cross",
    "obstacles_mix_circle",
    "end_of_lessons"
]
GOALS = get_goals(SCENES)
STARTS = get_starts(SCENES)

for scene in SCENES:
    plot_scene(scene, STARTS[scene], GOALS[scene])
    plot_scene_traj("sbp", scene, 0, GOALS[scene])

plt.show()
