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
VIZ_SAMPLE = 50


def sample_controls(means: torch.Tensor, covars: torch.Tensor,
                    sample_shape: torch.Size= torch.Size([])) -> torch.Tensor:
    """
    Sample gaussian controls from given means and covars.
    Used for plotting trajectories distribution as a comparison against particle methods.
    - Inputs:
        - means: (...,T,u_dim) tensor, mu
        - covars: (...,T,u_dim,T,u_dim) tensor, sigma
        - sample_shape: torch.Size, sample shape
    - Returns:
        - samples: (...,*sample_shape,T,u_dim)
    """
    batch_shape =  means.shape[:-2]
    T, u_dim = means.shape[-2:]
    samples = sample_gaussians(means.view(*batch_shape, T*u_dim), covars.view(*batch_shape, T*u_dim, T*u_dim),
                               sample_shape)
    return samples.view(*batch_shape, *sample_shape, T, u_dim)


def sample_gaussians(means: torch.Tensor, covars: torch.Tensor,
                     sample_shape: torch.Size= torch.Size([])):
    """
    Sample gaussians from given means and covars using torch Mutlivariate distribution.
    - Inputs:
        - means: (...,dim) tensor, mu
        - covars: (...,dim,dim) tensor, sigma
        - sample_shape: torch.Size, sample shape
    - Returns:
        - samples: (...,sample_shape, dim)
    """
    B = len(means.shape[:-1])
    S = len(sample_shape)
    distrb = torch.distributions.multivariate_normal.MultivariateNormal(means, covariance_matrix=covars)
    samples = distrb.rsample(sample_shape)
    return samples.permute(*[i for i in range(S, B+S)], *[i for i in range(S)], len(samples.shape)-1)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate visualizations for particles')

    parser.add_argument('-d', '--data-path', required=True, type=str,
                        help='Path to data location.')
    parser.add_argument('-s', '--scene', required=True, type=str,
                        help='Scene file associated with data.')
    parser.add_argument('-r', '--robot-radius', default=ROBOT_RADIUS, type=float,
                        help='Robot radius (meters).')
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
        means = np.load(os.path.join(args.data_path, f"means_{i:04d}.npy"))
        covs = np.load(os.path.join(args.data_path, f"covs_{i:04d}.npy"))
        means = torch.as_tensor(means, **args.tensor_kwargs)
        covs = torch.as_tensor(covs, **args.tensor_kwargs)
        sampled_controls = sample_controls(means, covs, (VIZ_SAMPLE,))
        state = states[i, :]

        traces = np.swapaxes(states[:(i + 1), :, :], 0, 1) if args.traces else None

        plt.cla()
        plt.clf()
        draw_belief_traj(plt.gca(), torch.as_tensor(state, **args.tensor_kwargs)[..., None, :],
                         sampled_controls, map_img, dmap.lims, args.goals,
                         vels=env.calc_vel_arrows(state, mag=1), traces=traces,
                         rollout_fn=robot_model.rollout,
                         robot_radius=ROBOT_RADIUS)
        plt.savefig(out_path)

    env.close()
