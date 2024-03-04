import time
from abc import ABC
from typing import Dict, Tuple, Iterable, Union
from statistics import mean
import torch
import torch_bp.bp as bp
from torch_bp.bp.linear_gbp import FactorGraph, LoopyLinearGaussianBP
from multi_robot_svbp.factors.linear_gaussian_base_factors import LinearGaussianUnaryRobotFactor, LinearGaussianPairwiseRobotFactor
from multi_robot_svbp.sim.robot import DynamicsModel, PrecomputedModel

class ControlInitializer(ABC):
    """
    Class used by CentralizedGBPController to initialize controls as a seed for LoopyGBP to start
    """
    def __call__(self, dt: float, horizon: int, dim: int,
                 current_states: torch.Tensor, tensor_kwargs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function signature used by CentralizedGBPController
        - Inputs:
            - dt: float, time difference between time steps in rollouts
            - horizon: int, number of time steps in rollouts
            - dim : int, base number of dimensions being controlled (x,y -> 2 dimensions)
            - current_states: (N, dim*2) current robot states (pose, vel)
        - Returns:
            - control_means: (N, T*dim) initial control means
            - control_covars: (N, T*dim, T*dim) initial control covriances
        """
        raise NotImplementedError

class DefaultInitializer(ControlInitializer):
    """
    Default method to generating new initial controls
    """
    def __init__(self, ctrl_perturb=.7, ctrl_sigma=.7) -> None:
        """
        - Inputs:
            - ctrl_perturb : float, small perturbation added when initializing new control means since setting mean to
                all zeros can cause singularity
            - ctrl_sigma: float, sigma for scaling the covariances of new initialized controls,
                such that new_covar = ctrl_sigma * eye
        """
        super().__init__()
        self.ctrl_perturb = ctrl_perturb
        self.ctrl_sigma = ctrl_sigma

    def __call__(self, dt: float, horizon: int, dim: int,
                 current_states: torch.Tensor, tensor_kwargs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
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
        control_means = torch.zeros(N, horizon*dim, **tensor_kwargs) + \
            self.ctrl_perturb * torch.randn(N,horizon*dim, **tensor_kwargs)
        control_covars = self.ctrl_sigma * torch.stack([torch.eye(horizon*dim, **tensor_kwargs) for _ in range(N)])

        return control_means, control_covars


class CentralizedGBPController(object):
    """
    Centralized controller using Linear Gaussian BP to control all agents
    """
    def __init__(self, N: int, adj_mat: torch.Tensor,
                 robot_models: Iterable[DynamicsModel],
                 robot_states: torch.Tensor,
                 factors: Iterable[Union[LinearGaussianUnaryRobotFactor, LinearGaussianPairwiseRobotFactor]],
                 factor_neighbours: Iterable[Iterable[int]],
                 dt=0.1, horizon=1, dim=2, init_msg_sigma=1e6,
                 ctrl_initializer: ControlInitializer= DefaultInitializer(),
                 tensor_kwargs={"device": "cpu", "dtype": torch.float64}):
        """
        Inputs:
        - N : int, number of agents (each agent action represented by a node)
        - adj_mat : bool tensor (N,N), indicating the connectivity between agents
        - robot_states : tensor (N,dim*2), all agent states (pose, vel)
        - factors : [LinearGaussianRobotFactor...], iterable of factor class that defines the
            relationship between node(s)
        - factor_neighbours : [tuple[int,..],..], iterable of tuple of ints, len corresponding to number of factors,
            each tuple represents the node ids that are connected to a given factor
        - dt : float, time difference between each projection step in rollout of agents
        - horizon : int, number of time project steps for the rollout
        - dim : int, base number of dimensions being controlled (x,y -> 2 dimensions)
        - init_msg_sigma : float, sigma values used for creating covariances for new messages
            suct that msg_covar = init_msg_sigma * I (assume values are uncorrelated to each other)
        - ctrl_initializer: ControlInitializer, intializer used to create initial controls to start the loopy
            Gaussian Belief Propagation
        - ctrl_perturb : float, small perturbation added when initializing new control means since setting mean to
            all zeros can cause singularity
        - ctrl_sigma: float, sigma for scaling the covariances of new initialized controls,
            such that new_covar = ctrl_sigma * eye
        - tensor_kwargs : dict, keyword args for tensor generation
        """
        self.N = N
        self.dt = dt
        self.T = horizon
        self.dim = dim
        self.init_msg_sigma = init_msg_sigma
        self.ctrl_initializer = ctrl_initializer
        self.tensor_kwargs = tensor_kwargs

        self.ctrl_iter = 0 # records number of control cycles ran

        # wrap with a precomputed model, allowing dynamics to be precomputed instead of calculating for
        # every factor
        self.robot_models = [PrecomputedModel(model) for model in robot_models]

        # initialize robot state and (control) nodes
        init_u_means, init_u_covars = ctrl_initializer(dt, horizon, dim, robot_states, tensor_kwargs)

        # precompute trajectories using models
        robot_states = robot_states.to(**tensor_kwargs)
        self.precompute_rollouts(init_u_means.view(-1,horizon,dim), robot_states)

        # store all factors, but only use active factors for solving
        self.all_factors = factors
        self.all_factor_neighbours = factor_neighbours
        self.adj_mat = adj_mat
        active_factors, active_factor_neighbours = self._filter_active_factors(adj_mat, factors, factor_neighbours)

        # update factor x0, robot model
        self.update_robot_states(robot_states)
        self._insert_model_to_factors()

        # init loopyGBP
        factor_graph = FactorGraph(N, active_factors, active_factor_neighbours)
        self._loopyGBP = LoopyLinearGaussianBP(init_u_means, init_u_covars, factor_graph, init_msg_sigma, tensor_kwargs)

    def solve(self, robot_states: torch.Tensor, adj_mat: torch.Tensor,
              msg_iters=1) -> torch.Tensor:
        """
        Solves the problem given the new updated states and adjacency matrix
        - Inputs:
            - robot_states: tensor (N,dim*2), all agent states (pose, vel)
            - adj_mat: bool tensor (N,N), indicating the connectivity between agents
            - msg_iters: int, number of message passing iterations to generate solution
        - Returns:
            - control_means: tensor (N,dim), mean of control solution for all agents (U)
            - control_covars: tensor (N,dim,dim), covar of control solution for all agents (U)
        """
        # update adjacency of nodes, robot states and reinitialize beliefs
        self.update_graph_connectivity(adj_mat)
        self.update_robot_states(robot_states)
        self.reinitialize_beliefs(robot_states)

        # actual message passing sequences
        for _ in range(msg_iters):
            start = time.time()

            # precompute rollouts and pass messages
            self.precompute_rollouts(self._loopyGBP.node_means.view(self.N, self.T, self.dim), robot_states)
            mean, covar = self._loopyGBP.solve(num_iters=1, msg_pass_per_iter=1)

        # print debug messages
        if self.ctrl_iter % 10 == 0:
            print("Timestep:", self.ctrl_iter)

        # for storing the number of ctrl cycles
        self.ctrl_iter += 1

        # return a clone mean and covars of final updated beliefs -> to prevent users from directly changing it
        return (mean.clone().view(self.N, self.T, self.dim),
                covar.clone().view(self.N, self.T, self.dim, self.T, self.dim))

    def get_current_beliefs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current control beliefs (without running solve)
        Returns:
            - control_means: tensor (N,T,dim), mean of control solution for all agents (U)
            - control_covars: tensor (N,T,dim,T,dim), covar of control solution for all agents (U)
        """
        # return a clone mean and covars of final updated beliefs -> to prevent users from directly changing it
        return (self._loopyGBP.node_means.clone().view(self.N, self.T, self.dim),
                self._loopyGBP.node_covars.clone().view(self.N, self.T, self.dim, self.T, self.dim))

    def update_graph_connectivity(self, adj_mat) -> None:
        """
        Updates the connectivity of the graph
        - Inputs:
            - adj_mat : bool tensor (N,N), indicating the connectivity between agents
        """
        if not torch.allclose(adj_mat, self.adj_mat):
            self.adj_mat = adj_mat
            active_factors, active_factor_neighbours = self._filter_active_factors(adj_mat,
                                                                                   self.all_factors,
                                                                                   self.all_factor_neighbours)
            # linear GBP does not have a method for resetting edges for now, hence we will just reinit a
            # new instance
            factor_graph = FactorGraph(self.N, active_factors, active_factor_neighbours)
            self._loopyGBP = LoopyLinearGaussianBP(self._loopyGBP.node_means, self._loopyGBP.node_covars,
                                                   factor_graph,
                                                   self.init_msg_sigma, self.tensor_kwargs)

    def precompute_rollouts(self, robot_controls: torch.tensor, robot_states: torch.Tensor) -> None:
        """
        Precomputes all the rollouts so that the cost incurred for evaluating factors dependent on them is only incurred
        once for each rollout trajectory
        - Inputs:
            - robot_controls: tensor (N,T,dim), all agent's controls (control U)
            - robot_states: tensor (N,dim*2), all agent's starting states (pose, vel)
        """
        for i, model in enumerate(self.robot_models):
            model.precompute_rollout_w_grad(robot_controls[i], robot_states[i])

    def update_robot_states(self, robot_states: torch.Tensor) -> None:
        """
        Update robot states by updating modules that are dependent on robot states
        - Inputs:
            - robot_states: tensor (N,dim*2), all agent's starting states (pose, vel)
        """
        for factor, factor_neighbour in zip(self.all_factors, self.all_factor_neighbours):
            factor.set_x_0(*[robot_states[i] for i in factor_neighbour])

    def reinitialize_beliefs(self, new_robot_states: torch.Tensor) -> None:
        """
        Reinitialize the beliefs for new planning cycle
        - Inputs:
            - new_robot_states: tensor (N,dim*2), all agent's starting states (pose, vel)
        """
        self._loopyGBP.node_means, self._loopyGBP.node_covars = self.ctrl_initializer(self.dt, self.T, self.dim,
                                                                                      new_robot_states,
                                                                                      self.tensor_kwargs)

    def _insert_model_to_factors(self) -> None:
        """
        Insert the model into the factors provided
        """
        for factor, factor_neighbour in zip(self.all_factors, self.all_factor_neighbours):
            factor.set_model(*[self.robot_models[i] for i in factor_neighbour])

    def _filter_active_factors(self, adj_mat: torch.Tensor,
                               factors: Iterable[Union[LinearGaussianUnaryRobotFactor, LinearGaussianPairwiseRobotFactor]],
                               factor_neighbours: Iterable[Iterable[int]]
                               ) -> Tuple[Iterable[Union[LinearGaussianUnaryRobotFactor, LinearGaussianPairwiseRobotFactor]],
                                          Iterable[Iterable[int]]]:
        """
        Takes in list of factors and their neighbours, return a filtered list of active factors depending on their
        connectivity
        - Inputs:
            - adj_mat: bool tensor (N,N), indicating the connectivity between agents
            - factors : [LinearGaussianRobotFactor...], iterable of factor class that defines the
                relationship between node(s)
            - factor_neighbours : [tuple[int,..],..], iterable of tuple of ints, len corresponding to number of factors,
                each tuple represents the node ids that are connected to a given factor
        """
        active_factors = []
        active_factor_neighbours = []
        for factor, factor_neighbour in zip(factors, factor_neighbours):
            if self._check_nodes_connected(adj_mat, factor_neighbour):
                active_factors.append(factor)
                active_factor_neighbours.append(factor_neighbour)

        return active_factors, active_factor_neighbours

    def _check_nodes_connected(self, adj_mat: torch.Tensor, nodes: Iterable[int]) -> torch.BoolTensor:
        """
        Query the adjacency matrix to see if the listed nodes are fully connected
        - Inputs:
            - adj_mat: (N,N) bool tensor, connectivity matrix
            - nodes: [int...] iterable, set of neighbours to check
        - NOTE:
            - for now assume bidirectional connectivity between all nodes is required
                for full connectivity
            - also assume all nodes will have connectivity to themselves
        """
        adj_mat = adj_mat + torch.eye(adj_mat.shape[-1], **self.tensor_kwargs)
        return torch.all(adj_mat[nodes, ...][..., nodes])
