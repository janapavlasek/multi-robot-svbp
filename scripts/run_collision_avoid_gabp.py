import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from multi_robot_svbp.envs import PointSwarm
from multi_robot_svbp.factors.linear_gaussian_trajectory_factors import LinearGaussianKBendingObstacleFactor, LinearGaussianTrajectoryCollisionFactor, LinearGaussianRunningCostJacFactor
from multi_robot_svbp.controllers.gbp_controller import ControlInitializer, CentralizedGBPController
from multi_robot_svbp.sim.robot import LinearPointRobotModel
from multi_robot_svbp.sim.map import DiffMap
from multi_robot_svbp.utils.plotting import draw_belief_traj
from multi_robot_svbp.utils.scenario_loader import load_default_scenario, load_yaml_scenario

FIG_WIDTH = 6
DT = 0.1
SIM_TIME = 5
HORIZON = 20
N = 5
COMM_RADIUS = 5.
SETPOINT_DIST = 0.5
MSG_PASS_ITERS = 50
MSG_SIGMA = 10.
CTRL_SIGMA = .7
CTRL_MAX_SPEED = 2.
CTRL_PERTURB = 1.2
ENV_PERTURB = 0  # 1e-4
VIZ_SAMPLE = 50
ROBOT_RADIUS = 0.2
COLLISION_TOL = 0.3  # 0.1


def create_robot_models(N: int, dt: float, horizon: int, tensor_kwargs: Dict):
    """
    Create robot model for each robot.
    Now every robot uses the same model.
    - Inputs:
        - N: int, number of robots
        - dt: float, time delta between time steps for robot rollouts
        - horizon: int, number of time steps for robot rollouts
        - tensor_kwargs: dict, keyword args to be used by tensors
    """
    return [LinearPointRobotModel(2, dt=dt, horizon=horizon, tensor_kwargs=tensor_kwargs)
            for _ in range(N)]


def create_factors(N: int, state: torch.Tensor, goals: torch.Tensor, dmap: DiffMap,
                   horizon: int, tensor_kwargs: Dict):
    """
    Create factor and factor neighbours for our problem.
    Use LinearGaussianRunningCostFactor, LinearGaussianObstacleFactor for every robot and
    TrajectoryCollisionFactor for every unique robot pair.
    - Inputs:
        - N: int, number of robots
        - state: (N, dim*2) tensor, current states of the robots
        - goals: (N, dim) tensor, goal states of the robots
        - dmap: DiffMap, DiffMap class used to handle signed distance field calculation
        - horizon: int, number of time steps for robot rollouts
        - tensor_kwargs: dict, keyword args to be used by tensors
    """

    # unary running cost jac factors
    unary_running_cost_factors = [LinearGaussianRunningCostJacFactor(
        torch.zeros(horizon,2, **tensor_kwargs), state[i],
        torch.zeros(horizon*6, **tensor_kwargs), 2.5 * torch.eye(horizon*6, **tensor_kwargs),
        c_pos=2., c_vel=2.5, c_u=2.5,
        horizon=horizon, goal=goals[i],
        alpha=1,
        tensor_kwargs=tensor_kwargs) for i in range(N)]
    unary_running_cost_factors_nbrs = [(i,) for i in range(N)]

    # unary obstacle factors
    unary_obs_factors = [LinearGaussianKBendingObstacleFactor(
        torch.zeros(horizon,2, **tensor_kwargs), state[i],
        torch.zeros(horizon, **tensor_kwargs), 1. * torch.eye(horizon, **tensor_kwargs),
        critical_dist=ROBOT_RADIUS + COLLISION_TOL, k_bend=1.8,
        sigma_T=torch.linspace(500., 450., horizon, **tensor_kwargs),
        horizon=horizon,
        signed_dist_map_fn=dmap.diff_map_fn,
        alpha=10.,
        tensor_kwargs=tensor_kwargs) for i in range(N)]
    unary_obs_factors_nbrs = [(i,) for i in range(N)]

    # trajectory collision factors
    traj_collision_factors = [LinearGaussianTrajectoryCollisionFactor(
        torch.zeros(horizon,2, **tensor_kwargs), state[i],
        torch.zeros(horizon,2, **tensor_kwargs), state[j],
        torch.zeros(1, **tensor_kwargs), 1e-3 * torch.eye(1, **tensor_kwargs),
        c_coll=300., c_coll_end=250.,
        horizon=horizon,
        r=2 * ROBOT_RADIUS + COLLISION_TOL, k=0.3,
        alpha=100.,
        tensor_kwargs=tensor_kwargs) for i in range(N) for j in range(i+1, N)]
    traj_collision_factors_nbrs = [(i,j) for i in range(N) for j in range(i+1, N)]

    # all factors
    factors = unary_running_cost_factors + unary_obs_factors + traj_collision_factors
    factor_nbrs = unary_running_cost_factors_nbrs + unary_obs_factors_nbrs + traj_collision_factors_nbrs

    return factors, factor_nbrs


class SimpleController(ControlInitializer):
    """
    Use a simple controller to determine the initial set of trajectories we want to optimize against
    - NOTE: you can use a different controller for your application if required
    """
    def __init__(self, goals: torch.Tensor, ctrl_sigma: float, max_speed: float,
                 ctrl_perturb: float) -> None:
        """
        - Inputs:
            - goals: (N, dim) tensor, goal positions to reach
            - ctrl_sigma: float, used to create control covars such that
                ctrl_covars = ctrl_sigma * I -> assume controls independent from each other
            - max_speed: float, max speed we want the robot to travel at in ms^-1
            - ctrl_perturb: float, small perturbation to break singularity
        """
        super().__init__()
        self.goals = goals
        self.ctrl_sigma = ctrl_sigma
        self.max_speed = max_speed
        self.ctrl_perturb = ctrl_perturb

    def __call__(self, dt: float, horizon: int, dim: int,
                 current_states: torch.Tensor, tensor_kwargs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Set speed by simply trying to move as much as possible in the direction of our goal
        - Inputs:
            - dt: float, time difference between time steps in rollouts
            - horizon: int, number of time steps in rollouts
            - dim : int, base number of dimensions being controlled (x,y -> 2 dimensions)
            - current_states: (N, dim*2) current robot states (pose, vel)
        - Returns:
            - control_means: (N, T*dim) initial control means
            - control_covars: (N, T*dim, T*dim) initial control covriances
        """
        N = current_states.shape[0]
        control_means = torch.zeros(N, horizon, dim, **tensor_kwargs)
        control_covars = self.ctrl_sigma * torch.eye(horizon*dim, **tensor_kwargs).expand(N, horizon*dim, horizon*dim)

        max_dist = self.max_speed * dt
        state = current_states.clone()
        for i in range(horizon):
            # find new pose to reach for this time step
            pose_delta = self.goals - state[...,:dim]
            dist = pose_delta.norm(dim=-1)
            limit_dist = dist > max_dist
            pose_delta[limit_dist] *= (max_dist / dist[limit_dist])[...,None]

            # find velocity req to reach new position
            vel_req = pose_delta / dt
            vel_delta = vel_req - state[...,dim:]

            # find acceleration req to reach new velocity
            control_means[:, i, ...] = vel_delta / dt

            # update state for next calculation
            state[...,:dim] += pose_delta
            state[...,dim:] = vel_req

        # perturb control to break singularity (especially if values are all zeros)
        control_means += self.ctrl_perturb * torch.randn(*control_means.shape, **tensor_kwargs)

        # reshape to form required by loopy GBP
        return control_means.view(N, horizon*dim), control_covars


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


def run_sim(args, out_dir="output/exp/gabp/collision", save_fig=True):
    # Load scenario
    scenario_data = load_default_scenario(args.num_robots, ENV_PERTURB, args.tensor_kwargs) \
        if args.scene is None else \
        load_yaml_scenario(args.scene, ENV_PERTURB, args.tensor_kwargs)

    # Setup environment
    num_robots = scenario_data['num_robots']
    dmap = DiffMap(scenario_data['map'], tensor_kwargs=args.tensor_kwargs)
    map_img = dmap.compute_binary_img().cpu()

    # Create robot models
    robot_models = create_robot_models(num_robots, DT, HORIZON, args.tensor_kwargs)

    # create sim environment
    env = PointSwarm(num_robots, dt=DT, comm_radius=args.comm_radius, vel_lim=3, cmd_lim=5,
                     lims=dmap.lims[:2], start_state=scenario_data['start_state'].cpu().numpy())
    steps = int(args.sim_time / DT)

    state, graph = env.get_state()
    if args.viz:
        env.render(edges=args.edges, vels=args.vels, save=True)

    # Setup factors, control initializer and controller
    goals = scenario_data['goals']
    factors, factor_nbrs = create_factors(num_robots, torch.tensor(state, **args.tensor_kwargs), goals,
                                          dmap, HORIZON, args.tensor_kwargs)
    # ctrl_intializer = SimpleController(goals, CTRL_SIGMA, CTRL_MAX_SPEED, CTRL_PERTURB)
    ctrl = CentralizedGBPController(num_robots, torch.tensor(graph, **args.tensor_kwargs),
                                    robot_models, torch.tensor(state, **args.tensor_kwargs),
                                    factors, factor_nbrs,
                                    DT, HORIZON, 2, MSG_SIGMA,
                                    # ctrl_intializer, # uncomment this to use own control initializer
                                    tensor_kwargs=args.tensor_kwargs)

    # Saving related variables
    # out_dir = f"output/exp/sbp/{args.scene.split('/')[-1].replace('.yml', '')}/{it}"
    os.makedirs(out_dir, exist_ok=True)

    # Plot starting
    means, covs = ctrl.get_current_beliefs()
    np.save(os.path.join(out_dir, f"means_0000.npy"), means.cpu().numpy())
    np.save(os.path.join(out_dir, f"covs_0000.npy"), covs.cpu().numpy())

    if save_fig:
        # Plotting related variables
        plt.figure(99, figsize=(FIG_WIDTH, FIG_WIDTH))
        plt.cla()
        plt.clf()

        sampled_controls = sample_controls(means, covs, (VIZ_SAMPLE,))
        draw_belief_traj(plt.gca(), torch.as_tensor(state, **args.tensor_kwargs)[..., None, :],
                         sampled_controls, map_img, dmap.lims, goals.cpu().numpy(),
                         vels=env.calc_vel_arrows(state, mag=1),
                         rollout_fn=[model.rollout for model in robot_models], robot_radius=ROBOT_RADIUS)
        out_path = os.path.join(out_dir, f"iteration_0000.jpg")
        plt.savefig(out_path)

    all_states = [state.copy()]
    all_ctrl = []

    # Simulate scenario
    for i in range(steps):
        u_means, u_covars = ctrl.solve(torch.tensor(state, **args.tensor_kwargs),
                                       torch.tensor(graph, **args.tensor_kwargs),
                                       MSG_PASS_ITERS)
        state, graph = env.step(u_means[:, 0, :].cpu().numpy())
        if args.viz:
            env.render(edges=args.edges, vels=args.vels, save=True)

        # Save the belief.
        np.save(os.path.join(out_dir, f"means_{i + 1:04d}.npy"), u_means.cpu().numpy())
        np.save(os.path.join(out_dir, f"covs_{i + 1:04d}.npy"), u_covars.cpu().numpy())

        if save_fig:
            plt.cla()
            plt.clf()
            sampled_controls = sample_controls(u_means, u_covars, (VIZ_SAMPLE,))
            draw_belief_traj(plt.gca(), torch.as_tensor(state, **args.tensor_kwargs)[..., None, :],
                             sampled_controls, map_img, dmap.lims, goals.cpu().numpy(),
                             vels=env.calc_vel_arrows(state, mag=1),
                             rollout_fn=[model.rollout for model in robot_models], robot_radius=ROBOT_RADIUS)
            out_path = os.path.join(out_dir, f"iteration_{i + 1:04d}.jpg")
            plt.savefig(out_path)

        all_states.append(state.copy())
        all_ctrl.append(u_means[:, 0, :].cpu().numpy().copy())

    all_states = np.stack(all_states)
    np.save(os.path.join(out_dir, "states.npy"), np.stack(all_states))

    all_ctrl = np.stack(all_ctrl)
    np.save(os.path.join(out_dir, "ctrl.npy"), np.stack(all_ctrl))

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Linearized GBP Collision Avoidance')
    
    parser.add_argument('-s', '--scene', default="data/scenes/squares_cross.yml", type=str,
                        help='Scenario file Path')
    parser.add_argument('-n', '--num-robots', default=N, type=int,
                        help='Number of agents. (Not applicable if scenario file was defined)')
    parser.add_argument('-k', '--num-particles', default=50, type=int,
                        help='Number of particles.')
    parser.add_argument('-t', '--sim-time', default=SIM_TIME, type=float,
                        help='Length of simulation (seconds).')
    parser.add_argument('-r', '--comm-radius', default=COMM_RADIUS, type=float,
                        help=f'Communication radius (meters). Default: {COMM_RADIUS} m')
    parser.add_argument('--robot-radius', default=ROBOT_RADIUS, type=float,
                        help=f'Robot radius (meters). Default: {ROBOT_RADIUS} m')
    parser.add_argument('--dt', default=DT, type=float,
                        help=f'Robot radius (meters). Default: {DT} secs')
    parser.add_argument('--ctrl-space', default='acc', type=str,
                        help='Control space. Default: \'acc\'')
    parser.add_argument('--runs', default=1, type=int,
                        help='Number of runs.')
    parser.add_argument('--viz', action='store_true',
                        help='Draw sim.')
    parser.add_argument('--edges', action='store_true',
                        help='Draw edges on sim.')
    parser.add_argument('--vels', action='store_true',
                        help='Draw velocity arrows on sim.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU.')
    parser.add_argument('--save', action='store_true',
                        help='Save figures.')

    args = parser.parse_args()

    device = "cuda" if args.cuda else "cpu"
    args.tensor_kwargs = {"device": device, "dtype": torch.float64}

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.runs == 1:
        run_sim(args)
    else:
        scene_file = args.scenario_file.split('/')[-1].replace('.yml', '')
        for i in range(args.runs):
            print("RUN", i)
            out_dir = f"output/exp/gabp/{scene_file}/{i}"
            run_sim(args, out_dir=out_dir, save_fig=args.save)
