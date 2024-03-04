import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multi_robot_svbp.envs import PointSwarm
from multi_robot_svbp.controllers.mpc import SteinMPC
from multi_robot_svbp.sim.robot import LinearPointRobotModel
from multi_robot_svbp.sim.map import DiffMap
from multi_robot_svbp.utils.plotting import draw_belief_traj

from utils.point_swarm import make_costs

FIG_WIDTH = 6
DT = 0.1
SIM_TIME = 5
HORIZON = 20


def parse_args():
    parser = argparse.ArgumentParser(description='Point Robot Stein MPC Controller.')

    parser.add_argument('-k', '--num-particles', default=50, type=int,
                        help='Number of particles.')
    parser.add_argument('--dt', default=DT, type=float,
                        help='Time step (seconds).')
    parser.add_argument('-t', '--sim-time', default=SIM_TIME, type=float,
                        help='Length of simulation (seconds).')
    parser.add_argument('--cuda', action='store_true',
                        help='Use the GPU as the tensor device.')

    args = parser.parse_args()

    device = "cuda" if args.cuda else "cpu"
    args.tensor_kwargs = {"device": device, "dtype": torch.float}

    return args


if __name__ == '__main__':
    args = parse_args()

    # torch.autograd.set_detect_anomaly(True)

    # Setup environment.
    dmap = DiffMap("data/squares.yml", tensor_kwargs=args.tensor_kwargs)
    map_img = dmap.compute_binary_img().cpu()

    state = torch.as_tensor([1, 1, 0, 0], **args.tensor_kwargs)  # Robot start state.
    goal = torch.as_tensor([9, 9], **args.tensor_kwargs)

    steps = int(args.sim_time / args.dt)

    # Model the robot motion.
    robot_model = LinearPointRobotModel(2, dt=args.dt, horizon=HORIZON, tensor_kwargs=args.tensor_kwargs)

    # Setup the costs for this controller.
    costs = make_costs(c_pos=0.1, c_vel=0.25, c_u=0.5, horizon=HORIZON, goal=goal,
                       signed_dist_map_fn=dmap.diff_map_fn,
                       # State bounds.
                       # pos_lims=[[0, 0], [10, 10]], c_pos_bounds=10,
                       max_vel=2, max_acc=3,
                       c_vel_bounds=10, c_acc_bounds=10,
                       tensor_kwargs=args.tensor_kwargs)
    # Create the controller.
    ctrl = SteinMPC(costs, args.num_particles, dt=args.dt, horizon=HORIZON, dim=2,
                    tensor_kwargs=args.tensor_kwargs)

    out_dir = "output/stein_mpc/"
    os.makedirs(out_dir, exist_ok=True)

    # Plot the initial state.
    out_path = os.path.join(out_dir, f"iteration_0000.jpg")
    plt.figure(99, figsize=(FIG_WIDTH, FIG_WIDTH))
    plt.cla()
    plt.clf()

    particles = ctrl.action_particles().contiguous()
    draw_belief_traj(plt.gca(), state[None, None, :], particles[None, ...],
                     map_img, dmap.lims, goal[None, ...],
                     rollout_fn=robot_model.rollout, vels=state.split(1))

    plt.savefig(out_path)

    for i in range(steps):
        if i % 10 == 0:
            print("Timestep", i)

        # Compute the best action using the controller.
        u = ctrl.solve(state, n_iters=30)[:, -2:].contiguous()

        # Apply this action.
        traj = robot_model.rollout(u, state)
        state = traj[0, :4]  # Make the current state equal to the next element in the trajectory.

        # Plot the trajectories.
        out_path = os.path.join(out_dir, f"iteration_{i + 1:04d}.jpg")
        particles = ctrl.action_particles().contiguous()

        plt.cla()
        plt.clf()
        draw_belief_traj(plt.gca(), state[None, None, :], particles[None, ...],
                         map_img, dmap.lims, goal[None, ...],
                         rollout_fn=robot_model.rollout, vels=state.split(1))
        plt.savefig(out_path)
        plt.pause(0.001) # update image online

    print("Complete")
    plt.show() # leave image on screen
