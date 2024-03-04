import os
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multi_robot_svbp.envs import PointSwarm
from multi_robot_svbp.sim.robot import LinearPointRobotModel
from multi_robot_svbp.sim.map import DiffMap
from multi_robot_svbp.utils.plotting import draw_belief_traj


FIG_WIDTH = 6
DT = 0.1
SIM_TIME = 5
HORIZON = 20
N = 5
COMM_RADIUS = 10
SETPOINT_DIST = 0.5
MSG_PASS_ITERS = 25
MSG_PASS_UPDATES = 25
ROBOT_RADIUS = 0.2
COLLISION_TOL = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description='Generate visualizations for particles')

    parser.add_argument('-d', '--data-path', required=True, type=str,
                        help='Path to data location.')
    parser.add_argument('-s', '--scene', required=True, type=str,
                        help='Scene file associated with data.')
    parser.add_argument('-r', '--robot-radius', default=ROBOT_RADIUS, type=float,
                        help='Robot radius (meters).')
    parser.add_argument('--collision-radius', default=None, type=float,
                        help='Collision radius to display (meters).')
    parser.add_argument('--no-dist', action='store_true',
                        help='Do not plot the distribution.')
    parser.add_argument('--traces', action='store_true',
                        help='Draw the path trace so far.')

    args = parser.parse_args()

    args.tensor_kwargs = {"device": "cpu", "dtype": torch.float}

    with open(args.scene, "r") as f:
        data = yaml.load(f, Loader=yaml.Loader)

    args.num_robots = data["num_agents"]
    args.map_file = data["map"]
    args.starts = np.array(data["starts"])
    args.goals = np.array(data["goals"])

    return args


if __name__ == '__main__':
    args = parse_args()

    # Setup environment.
    dmap = DiffMap(args.map_file, tensor_kwargs=args.tensor_kwargs)
    map_img = dmap.compute_binary_img().cpu()

    # Load the states.
    states = np.load(os.path.join(args.data_path, "states.npy"))
    iters, num_robots, _ = states.shape

    starts = np.concatenate([args.starts, np.zeros((args.num_robots, 2))], axis=1)

    # Needed for plotting and gradients for the factors.
    robot_model = LinearPointRobotModel(2, dt=DT, horizon=HORIZON, tensor_kwargs=args.tensor_kwargs)
    env = PointSwarm(args.num_robots, dt=DT,
                     lims=dmap.lims[:2], start_state=starts)

    plt.figure(99, figsize=(FIG_WIDTH, FIG_WIDTH))
    for i in range(iters - 1):
        out_path = os.path.join(args.data_path, f"iteration_{i:04d}.jpg")
        if not args.no_dist:
            particles = np.load(os.path.join(args.data_path, f"particles_{i:04d}.npy"))
            particles = torch.as_tensor(particles, **args.tensor_kwargs)
        else:
            particles = None
        state = states[i, :]

        traces = np.swapaxes(states[:(i + 1), :, :], 0, 1) if args.traces else None

        plt.cla()
        plt.clf()
        draw_belief_traj(plt.gca(), torch.as_tensor(state, **args.tensor_kwargs)[..., None, :],
                         particles, map_img, dmap.lims, args.goals, vels=env.calc_vel_arrows(state, mag=1),
                         rollout_fn=robot_model.rollout, traces=traces,
                         robot_radius=args.robot_radius, collision_radius=args.collision_radius)
        plt.savefig(out_path)

    env.close()
