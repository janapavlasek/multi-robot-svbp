import os
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multi_robot_svbp.envs import PointSwarm
from multi_robot_svbp.factors.trajectory_factors import UnaryRobotTrajectoryFactor, TrajectoryCollisionFactor
from multi_robot_svbp.controllers.svbp_controller import CentralizedSVBPController
from multi_robot_svbp.sim.robot import LinearPointRobotModel
from multi_robot_svbp.sim.map import DiffMap
from multi_robot_svbp.utils.plotting import draw_belief_traj

from utils.point_swarm import make_costs

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
COLLISION_TOL = 0.3  # 0.1


def create_factors(args, robot_model, state, dmap, goals):
    goals = torch.as_tensor(goals, **args.tensor_kwargs)
    state = torch.as_tensor(state, **args.tensor_kwargs)
    # Get the gradients for the robot model. Grads don't depend on state for linear model.
    tmp_u = torch.normal(0, 1, size=(args.num_robots, args.num_particles, HORIZON, 2), **args.tensor_kwargs)
    state_grads, _, _ = robot_model.rollout_w_grad(tmp_u, state[..., None, :])
    edge_factors = TrajectoryCollisionFactor(c_coll=500, c_coll_end=50,
                                             horizon=HORIZON, ctrl_space=args.ctrl_space,
                                             r=args.robot_radius * 2 + COLLISION_TOL,
                                             traj_grads_U_s=state_grads[0, 0, ...],
                                             traj_grads_U_t=state_grads[0, 0, ...],
                                             tensor_kwargs=args.tensor_kwargs)

    unary_factors = []
    for i in range(args.num_robots):
        costs = make_costs(c_pos=0.1, c_vel=0.35, c_u=0.5, c_term=6, obs_avoid_dist=args.robot_radius + COLLISION_TOL,
                           ctrl_space=args.ctrl_space, horizon=HORIZON, goal=goals[i],
                           signed_dist_map_fn=dmap.diff_map_fn,
                           # State bounds.
                           # pos_lims=[[0, 0], [10, 10]], c_pos_bounds=10,
                           max_vel=2, max_acc=3,
                           c_vel_bounds=10, c_acc_bounds=10,
                           tensor_kwargs=args.tensor_kwargs)
        factor = UnaryRobotTrajectoryFactor(costs, horizon=HORIZON, ctrl_space=args.ctrl_space,
                                            traj_grads_U=state_grads[i, 0, ...],
                                            tensor_kwargs=args.tensor_kwargs)
        unary_factors.append(factor)

    return unary_factors, edge_factors


def run_sim(args, it=0, save_fig=True):
    # Setup environment.
    dmap = DiffMap(args.map_file, tensor_kwargs=args.tensor_kwargs)
    map_img = dmap.compute_binary_img().cpu()

    # Add zero velocities to start.
    starts = args.starts
    if args.ctrl_space == 'acc':
        starts = np.concatenate([starts, np.zeros((args.num_robots, 2))], axis=1)

    env = PointSwarm(args.num_robots, dt=DT, comm_radius=args.comm_radius, ctrl_space=args.ctrl_space,
                     vel_lim=3, cmd_lim=5, lims=dmap.lims[:2], start_state=starts)
    steps = int(args.sim_time / DT)

    state, graph = env.get_state()
    if args.viz:
        env.render(edges=args.edges, vels=args.vels, save=True)

    # Needed for plotting and setting factor gradients.
    robot = LinearPointRobotModel(2, dt=args.dt, horizon=HORIZON, ctrl_space=args.ctrl_space,
                                  tensor_kwargs=args.tensor_kwargs)

    # Setup the factors.
    node_factors, edge_factors = create_factors(args, robot, state, dmap, args.goals)
    ctrl = CentralizedSVBPController(args.num_robots, graph, state, node_factors, edge_factors,
                                     ctrl_space=args.ctrl_space, goals=args.goals,
                                     num_particles=args.num_particles, dt=DT, horizon=HORIZON,
                                     tensor_kwargs=args.tensor_kwargs)

    out_dir = f"output/exp/sbp/{args.scene.split('/')[-1].replace('.yml', '')}/{it}"
    os.makedirs(out_dir, exist_ok=True)

    particles = ctrl.particles()[:, :, :, -2:].contiguous()
    np.save(os.path.join(out_dir, "particles_0000.npy"), particles.cpu().numpy())

    if save_fig:
        plt.figure(99, figsize=(FIG_WIDTH, FIG_WIDTH))
        out_path = os.path.join(out_dir, f"iteration_0000.jpg")

        plt.cla()
        plt.clf()

        vel_states = np.concatenate((state, np.zeros_like(state)), axis=-1) if args.ctrl_space == 'vel' else state
        draw_belief_traj(plt.gca(), torch.as_tensor(state, **args.tensor_kwargs)[..., None, :], particles,
                         map_img, dmap.lims, args.goals, robot_radius=args.robot_radius,
                         vels=env.calc_vel_arrows(vel_states, mag=1), rollout_fn=robot.rollout)

        plt.savefig(out_path)

    all_states = [state.copy()]
    all_ctrl = []
    for i in range(steps):
        u = ctrl.solve(state, graph, msg_iters=1, particle_iters=30)
        state, graph = env.step(u)

        all_states.append(state.copy())
        all_ctrl.append(u.copy())

        particles = ctrl.particles()[:, :, :, -2:].contiguous()
        np.save(os.path.join(out_dir, f"particles_{i + 1:04d}.npy"), particles.cpu().numpy())

        if args.viz:
            env.render(edges=args.edges, vels=args.vels, save=True)

        if save_fig:
            out_path = os.path.join(out_dir, f"iteration_{i + 1:04d}.jpg")
            plt.cla()
            plt.clf()
            vel_states = np.concatenate((state, u), axis=-1) if args.ctrl_space == 'vel' else state
            draw_belief_traj(plt.gca(), torch.as_tensor(state, **args.tensor_kwargs)[..., None, :], particles,
                             map_img, dmap.lims, args.goals, robot_radius=args.robot_radius,
                             vels=env.calc_vel_arrows(vel_states, mag=1), rollout_fn=robot.rollout)
            plt.savefig(out_path)

    all_states = np.stack(all_states)
    np.save(os.path.join(out_dir, "states.npy"), np.stack(all_states))

    all_ctrl = np.stack(all_ctrl)
    np.save(os.path.join(out_dir, "ctrl.npy"), np.stack(all_ctrl))

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='BP Flocking')

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
    args.tensor_kwargs = {"device": device, "dtype": torch.float}

    with open(args.scene, "r") as f:
        data = yaml.load(f, Loader=yaml.Loader)

    args.num_robots = data["num_agents"]
    args.map_file = data["map"]
    args.starts = np.array(data["starts"])
    args.goals = np.array(data["goals"])

    assert args.starts.shape[0] == args.num_robots, "Must provide the same number of starts as robots."
    assert args.goals.shape[0] == args.num_robots, "Must provide the same number of goals as robots."

    return args


if __name__ == '__main__':
    args = parse_args()

    # import torch
    # torch.autograd.set_detect_anomaly(True)

    for it in range(args.runs):
        print("RUN", it)
        run_sim(args, it, save_fig=args.save)
