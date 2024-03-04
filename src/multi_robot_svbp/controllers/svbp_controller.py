import time
import numpy as np

import torch

import torch_bp.bp as bp
from torch_bp.graph import MRFGraph
from torch_bp.inference.kernels import RBFMedianKernel
from torch_bp.util.distances import pairwise_euclidean_distance

from multi_robot_svbp.sim.robot import LinearPointRobotModel
from multi_robot_svbp.utils.distance import TrajectoryDistance, euclidean_path_length


class SVBPSwarmController(object):
    def __init__(self, N, adj_mat, init_particles, node_factors=None, edge_factors=None,
                 num_particles=50, dt=0.1, horizon=1, dim=2, ctrl_space='acc', nodes_to_solve=None,
                 optim_params={"lr": 0.05}, tensor_kwargs={"device": "cpu", "dtype": torch.float}):
        self.K = num_particles
        self.N = N
        self.dt = dt
        self.horizon = horizon
        self.dim = dim

        if ctrl_space == 'acc':
            self.p_dim = dim * 3
        elif ctrl_space == 'vel':
            self.p_dim = dim * 2
        else:
            raise Exception(f"Unrecognized control space: {ctrl_space}")

        self.ctrl_space = ctrl_space
        self.optim_params = optim_params
        self.tensor_kwargs = tensor_kwargs

        self.adj_mat = adj_mat.copy()
        self.graph = None
        self.sbp = None
        self.optim = None
        self.ctrl_iter = 0

        # NOTE: only allows linear models for now, for non-linear models we have to change the control flow!!
        self.robot_model = LinearPointRobotModel(dim, dt=dt, horizon=self.horizon, ctrl_space=ctrl_space,
                                                 tensor_kwargs=tensor_kwargs)

        gamma = 1. / np.sqrt(2 * self.horizon * self.dim)
        rbf_kernel = RBFMedianKernel(gamma=gamma, distance_fn=TrajectoryDistance(self.dim, self.horizon))
        # gammas = [rbf_kernel.median_heuristic_svgd(init_particles[i, ...].view(self.K, self.horizon * self.p_dim))
        #           for i in range(N)]
        # rbf_kernel.gamma = np.mean(gammas)

        # Graph and solver.
        self.graph = self.init_graph(adj_mat, node_factors, edge_factors)
        self.sbp = bp.LoopySVBP(init_particles.view(self.N, self.K, self.horizon * self.p_dim),
                                self.graph, rbf_kernel, msg_init_mode="pairwise", nodes_to_solve=nodes_to_solve)
        self.init_optimizer()

    def reset_graph(self, adj_mat, node_factors, edge_factors):
        self.adj_mat = adj_mat.copy()
        graph = self.init_graph(adj_mat, node_factors, edge_factors)
        self.graph = graph
        self.sbp.graph = graph

    def init_graph(self, adj_mat, node_factors, edge_factors):
        # Compute the list of edges from the adjacency matrix.
        edges = np.stack(np.triu(adj_mat).nonzero(), axis=1)

        # If there are multiple edge factors, ensure there is one per edge.
        if isinstance(edge_factors, list):
            assert len(edge_factors) == edges.shape[0], "Must provide same number of edge factors as edges."
        # If there are multiple node factors, ensure there is one per node.
        if isinstance(node_factors, list):
            assert len(node_factors) == self.N, "Must provide same number of node factors as nodes."

        self.graph
        return MRFGraph(self.N, edges, edge_factors=edge_factors, unary_factors=node_factors)

    def init_optimizer(self):
        self.optim = torch.optim.Adam(self.sbp.optim_parameters(), **self.optim_params)

    def rollout(self, state, action_seq):
        return self.robot_model(action_seq, state)

    def particles(self):
        return self.sbp.particles().view(self.N, self.K, self.horizon, self.p_dim)

    def pass_messages(self, msg_iters=1, precompute=False):
        with torch.no_grad():
            for _ in range(msg_iters):
                self.sbp.pass_messages(normalize=True, precompute=precompute)


class CentralizedSVBPController(SVBPSwarmController):
    def __init__(self, N, adj_mat, states, node_factors=None, edge_factors=None, goals=None,
                 num_particles=50, dt=0.1, horizon=1, dim=2, init_cov=0.5, ctrl_space='acc',
                 optim_params={"lr": 0.01}, tensor_kwargs={"device": "cpu", "dtype": torch.float}):
        # Initialize all the nodes. The positions should be the same and the accelerations are random.
        states = torch.as_tensor(states, **tensor_kwargs)
        init_u = torch.normal(0, init_cov, size=(N, num_particles, horizon, dim)).to(**tensor_kwargs)
        robot_model = LinearPointRobotModel(dim, dt=dt, horizon=horizon, ctrl_space=ctrl_space,
                                            tensor_kwargs=tensor_kwargs)
        init_particles = robot_model.rollout(init_u, states[..., None, :])

        self.goals = torch.as_tensor(goals, **tensor_kwargs)
        self.init_cov = init_cov

        super().__init__(N, adj_mat, init_particles, node_factors=node_factors, edge_factors=edge_factors,
                         num_particles=num_particles, dt=dt, horizon=horizon, dim=dim, ctrl_space=ctrl_space,
                         optim_params=optim_params, tensor_kwargs=tensor_kwargs)

    def rollout_all(self, states):
        u = self.particles()[:, :, :, -self.dim:]
        particles = self.robot_model.rollout(u.contiguous(), states[..., None, :])
        # NOTE: we did not re-evaluate gradients since we assumed model to be linear hence traj grad is const to U
        self.sbp.reset(particles.view(self.N, self.K, self.horizon * self.p_dim))

    def roll_particles(self, states, shift_steps=1):
        u = self.sbp.particles().view(self.N, self.K, self.horizon, self.p_dim)[:, :, :, -self.dim:]
        u = u.roll(-shift_steps, 2)
        new_term = torch.zeros(1, **self.tensor_kwargs)
        u[:, :, -shift_steps:, :] = new_term
        particles = self.robot_model.rollout(u, states[..., None, :])
        self.sbp.reset(particles.view(self.N, self.K, self.horizon * self.p_dim))

    def detect_convergence(self, particles, states):
        was_reset = False
        for i in range(self.N):
            ave_path_len = euclidean_path_length(particles[i, :, :, :self.dim]).mean()
            ave_goal_dist = pairwise_euclidean_distance(particles[i, :, -1, :self.dim], self.goals[i], squared=False).mean()

            if ave_path_len < 0.35 and ave_goal_dist > 0.5:
                init_u = torch.normal(0, self.init_cov, size=(self.K, self.horizon, self.dim),
                                      **self.tensor_kwargs)
                particles[i, :, :, -self.dim:] = init_u
                was_reset = True
                print("\trobot reset", i, ave_path_len.item(), ave_goal_dist.item())
        if was_reset:
            self.sbp.reset(particles.view(self.N, self.K, self.horizon * self.p_dim))
            self.rollout_all(states)

    def solve(self, states, adj_mat, msg_iters=1, particle_iters=10, reset=True):
        states = torch.as_tensor(states, **self.tensor_kwargs)

        # If the graph has changed, reset it.
        if not np.allclose(self.adj_mat, adj_mat):
            self.reset_graph(adj_mat, self.graph.unary_factors, self.graph.edge_factors)

        self.sbp.reset_msgs()

        # Initialize the optimizer.
        if self.ctrl_iter > 0:
            self.roll_particles(states)
        self.init_optimizer() #TODO this is intergrated to the solve function

        if reset and self.ctrl_iter > 0:
            particles = self.sbp.particles().view(self.N, self.K, self.horizon, self.p_dim)
            self.detect_convergence(particles, states)

        if self.ctrl_iter % 10 == 0:
            print("Timestep:", self.ctrl_iter)

        precompute_times = []
        pass_msg_times = []
        update_times = []

        for i in range(particle_iters):
            start = time.time()

            if i > 0:
                # SVBP will only update the control signal, since the factors
                # guarantee that the gradients for the rest of the state is zero.
                # This resets the state by rolling out the control signal.
                self.rollout_all(states)

            # Precompute the factors before running this loop.
            self.sbp.precompute_pairwise()
            self.sbp.precompute_unary()
            precompute_times.append(time.time() - start)

            start = time.time()
            self.pass_messages(msg_iters, precompute=False)
            pass_msg_times.append(time.time() - start)

            start = time.time()

            self.optim.zero_grad()
            self.sbp.update(False)  # Don't recompute messages
            self.optim.step()

            update_times.append(time.time() - start)

        print(f"\t precompute: {np.mean(precompute_times):.4f} (Total: {np.sum(precompute_times):.4f}) "
              f"msg updates: {np.mean(pass_msg_times):.4f} (Total: {np.sum(pass_msg_times):.4f}) "
              f"step particles: {np.mean(update_times):.4f} (Total: {np.sum(update_times):.4f})")

        # Get the best particle for each node.
        ctrls = []
        self.sbp.pass_messages(precompute=True)
        for n in range(self.N):
            weights = self.sbp.compute_belief_weights(n, recompute_factors=False)
            ctrl_mle = self.sbp.particles(n)[weights.argmax()].view(self.horizon, self.p_dim)
            ctrls.append(ctrl_mle[0, -self.dim:].detach().cpu().numpy())

        self.ctrl_iter += 1

        return np.stack(ctrls)


class DecentralizedSVBPController(SVBPSwarmController):
    def __init__(self, N, adj_mat, state, self_idx=0, node_factors=None, edge_factors=None,
                 num_particles=50, dt=0.1, horizon=1, dim=2, init_cov=0.5, ctrl_space='acc',
                 optim_params={"lr": 0.01}, tensor_kwargs={"device": "cpu", "dtype": torch.float}):
        self.self_idx = self_idx
        self.init_cov = init_cov

        # Initialize just this node.
        state = torch.as_tensor(state, **tensor_kwargs)
        init_u = torch.normal(0, init_cov, size=(num_particles, horizon, dim)).to(**tensor_kwargs)
        robot_model = LinearPointRobotModel(dim, dt=dt, horizon=horizon, ctrl_space=ctrl_space,
                                            tensor_kwargs=tensor_kwargs)
        init_particles = robot_model.rollout(init_u, state[None, :])
        K, T, p_dim = init_particles.shape
        particles = torch.zeros(N, K, T, p_dim, **tensor_kwargs)
        particles[self.self_idx] = init_particles

        super().__init__(N, adj_mat, particles, node_factors=node_factors, edge_factors=edge_factors,
                         num_particles=num_particles, dt=dt, horizon=horizon, dim=dim, ctrl_space=ctrl_space,
                         solve_node=self.self_idx, optim_params=optim_params, tensor_kwargs=tensor_kwargs)

    def reset_particles(self, state, nbr_particles=None, nbr_idx=None, shift_steps=0):
        # Grab all the current particles.
        particles = self.sbp.particles().view(self.N, self.K, self.horizon, self.p_dim)
        u = particles[self.self_idx, :, :, -self.dim:]  # Actions for this robot.
        if shift_steps > 0:
            # Shift this robot's actions.
            u = u.roll(-shift_steps, -2)
            new_term = torch.normal(0, self.init_cov, size=(self.K, shift_steps, self.dim)).to(**self.tensor_kwargs)
            u[:, -shift_steps:, :] = new_term
        # Rollout the current particles.
        self_particles = self.robot_model.rollout(u.contiguous(), state[..., None, :])
        particles[self.self_idx] = self_particles

        # If nbr particles were given, reset those.
        if nbr_particles is not None:
            num_nbrs = len(nbr_idx)
            nbr_particles = nbr_particles.view(num_nbrs, self.K, self.horizon, self.p_dim)
            particles[nbr_idx] = nbr_particles.to(**self.tensor_kwargs)

        self.sbp.reset(particles.view(self.N, self.K, self.horizon * self.p_dim))

    def particles(self):
        particles = self.sbp.particles(self.self_idx)
        return particles

    def solve(self, state, adj_mat, nbr_particles=None, nbr_idx=None, msg_iters=1, particle_iters=10, shift_steps=0):
        state = torch.as_tensor(state, **self.tensor_kwargs).unsqueeze(0)
        if nbr_particles is not None:
            nbr_particles = torch.as_tensor(nbr_particles, **self.tensor_kwargs)

        start = time.time()
        # If the graph has changed, reset it.
        if not np.allclose(self.adj_mat, adj_mat):
            self.reset_graph(adj_mat, self.graph.unary_factors, self.graph.edge_factors)

        self.sbp.reset_msgs()

        # Initialize the optimizer.
        self.reset_particles(state, nbr_particles=nbr_particles, nbr_idx=nbr_idx, shift_steps=shift_steps)
        self.init_optimizer()

        if self.ctrl_iter % 10 == 0:
            print("Timestep:", self.ctrl_iter)

        # Precompute the factors.
        self.sbp.precompute_unary()
        self.sbp.precompute_pairwise()

        print(f"\tInit time: {time.time() - start:.4f}")

        pass_msg_times = []
        update_times = []

        for i in range(particle_iters):
            start = time.time()

            # Precompute the factors before running this loop, just for this particle.
            if i > 0:
                self.sbp.precompute_pairwise_single(self.self_idx)
                self.sbp.precompute_unary_single(self.self_idx)

            self.pass_messages(msg_iters, precompute=False)
            pass_msg_times.append(time.time() - start)

            start = time.time()

            self.optim.zero_grad()
            self.sbp.update(False)  # Don't recompute messages
            self.optim.step()

            update_times.append(time.time() - start)

            # SVBP will only update the control signal, since the factors
            # guarantee that the gradients for the rest of the state is zero.
            # This resets the state by rolling out the control signal. Make
            # sure the robot particles other than this robot's are unchanged.
            self.reset_particles(state, shift_steps=0)

        print(f"\t msg updates: {np.mean(pass_msg_times):.4f} (Total: {np.sum(pass_msg_times):.4f}) "
              f"step particles: {np.mean(update_times):.4f} (Total: {np.sum(update_times):.4f})")

        # Get the best particle for each node.
        self.sbp.precompute_unary_single(self.self_idx)
        self.sbp.precompute_pairwise()
        self.sbp.pass_messages(precompute=False)
        weights = self.sbp.compute_belief_weights(self.self_idx, recompute_factors=False)
        ctrl_mle = self.sbp.particles(self.self_idx)[weights.argmax()].view(self.horizon, self.p_dim)

        self.ctrl_iter += 1

        return ctrl_mle.detach().cpu().numpy()
