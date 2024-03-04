"""
Trajectory factors to be used for standard robot collision avoidance in mapped environment scenarios
"""

from typing import Tuple
import torch

# import torch_bp.bp as bp
from torch_bp.graph.factors import UnaryFactor, PairwiseFactor

from multi_robot_svbp.costs.base_costs import CompositeSumCost
from multi_robot_svbp.costs.trajectory_costs import RunningCrossCollisionCost


class UnaryRobotTrajectoryFactor(UnaryFactor):
    def __init__(self, costs, dim=2, horizon=1, ctrl_space='acc',
                 optimize=False, traj_grads_U=None,  # trajectory gradients wrt to U
                 tensor_kwargs={"device": "cpu", "dtype": torch.float}):

        super().__init__()
        self.dim = dim
        if ctrl_space == 'acc':
            self.p_dim = dim * 3
        elif ctrl_space == 'vel':
            self.p_dim = dim * 2
        else:
            raise Exception(f"Unrecognized control space: {ctrl_space}")

        self.horizon = horizon
        self.tensor_kwargs = tensor_kwargs
        self.combined_cost = CompositeSumCost(costs=costs, sigma=-1,
                                              tensor_kwargs=tensor_kwargs)
        if traj_grads_U is not None:
            self.set_traj_grads(traj_grads_U)

        self.grad_log_likelihood = torch.compile(self._grad_log_likelihood) if optimize else self._grad_log_likelihood
        self.log_likelihood = torch.compile(self._log_likelihood) if optimize else self._log_likelihood

    def set_traj_grads(self, traj_grads_U) -> None:
        """
        - Inputs:
            - traj_grads_U: T,(x_dim+u_dim),T,u_dim) tensor wrt U
        """
        # NOTE: current implementation assumes trajectory wrt U is linear hence we do not update traj grads wrt to U
        x_dim = self.p_dim - self.dim
        self.traj_grads = torch.cat(
            (torch.zeros(self.horizon, self.p_dim, self.horizon, x_dim, **self.tensor_kwargs),
                traj_grads_U),
            dim=-1)
        # traj_grads -> (T,(x_dim+u_dim),T,(x_dim+u_dim)) wrt traj
        self.traj_grads = self.traj_grads.view(self.horizon * self.p_dim, self.horizon * self.p_dim)
        # traj_grads -> (T*(x_dim+u_dim),T*(x_dim+u_dim)) wrt traj

    def _log_likelihood(self, x) -> torch.Tensor:
        """
        - Inputs:
            - x: (...,T*dim*3) tensor, full trajectory state
        - Returns:
            - cost: (...,) tensor, scalar cost for each trajectory
        """
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.horizon, self.p_dim)
        return self.combined_cost.cost(x)

    def _grad_log_likelihood(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Inputs:
            - x: (...,T*dim*3) tensor, full trajectory state
        - Returns:
            - grad_cost: (...,T*(x_dim+u_dim)) tensor, wrt to stacked trajectory form
            - cost: (...,) tensor, scalar cost for each trajectory
        """
        # NOTE: current implementation assumes trajectory wrt U is linear hence we do not update traj grads wrt to U
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.horizon, self.p_dim)
        traj_cost_grad, log_px = self.combined_cost.grad_w_cost(x)
        # dc/d(U) = dc/d(X_bar) @ d(X_bar)/d(U)
        grad_log_px = (traj_cost_grad.view(*batch_shape, 1, self.horizon *
                                           (self.p_dim)) @ self.traj_grads)[..., 0, :]
        return grad_log_px, log_px


class TrajectoryCollisionFactor(PairwiseFactor):
    def __init__(self, optimize=False, c_coll=100., c_coll_end=None,
                 horizon=1, dim=2, r=0.5, k=0.3, ctrl_space='acc',
                 traj_grads_U_s=None, traj_grads_U_t=None,  # trajectory gradients wrt to U_s and U_t
                 tensor_kwargs={"device": "cpu", "dtype": torch.float}, **kwargs):
        """
        - Inputs:
            - c_coll: float, scalar cost for colliding with another robot
            - c_coll_end: None | float, if defined, changes c_coll to scale accross rollout linearly till furthest
                rollout projection c_coll value equals to c_coll_end
            - dim: int, dimension of workspace
            - horizon: int, length of projection of rollouts
            - r: float, radius of agents below which they are in collision
            - k: float, linearity bending factor, controls the curve of the function
                1: linear, >1: convex, <1: concave
            - traj_grads_U_s: None | (T,(x_dim+u_dim),T,u_dim) tensor, gradient of trajectory wrt to U for node s
            - traj_grads_U_t: None | (T,(x_dim+u_dim),T,u_dim) tensor, gradient of trajectory wrt to U for node t
            - tensor_kwargs: dict, keyword args used for creating new tensors
            - kwargs: keyword args, to pass to Pairwise factor
        """
        super().__init__(**kwargs)
        self.horizon = horizon  # Number of steps in horizon.
        self.dim = dim  # Dimension of the robot.
        if ctrl_space == 'acc':
            self.p_dim = dim * 3
        elif ctrl_space == 'vel':
            self.p_dim = dim * 2
        else:
            raise Exception(f"Unrecognized control space: {ctrl_space}")

        self.tensor_kwargs = tensor_kwargs
        if c_coll_end is not None:
            c_coll = torch.linspace(c_coll, c_coll_end, horizon, **tensor_kwargs)
        else:
            c_coll = c_coll
        self.cross_collision_cost_fn = RunningCrossCollisionCost(
            pos_dim=dim, radius=r, k_bend=k, sigma_T=c_coll, sigma=-1,
            tensor_kwargs=tensor_kwargs)
        self.set_traj_grad_s(traj_grads_U_s)
        self.set_traj_grad_t(traj_grads_U_t)

        self.grad_log_likelihood = torch.compile(self._grad_log_likelihood) if optimize else self._grad_log_likelihood
        self.log_likelihood = torch.compile(self._log_likelihood) if optimize else self._log_likelihood

    def _propagate_incoming_grad(self, incoming_grad):
        """
        Propagate incoming grad from U domain to traj domain
        - Inputs:
            - incoming_grad: (T,(x_dim+u_dim),T,u_dim) tensor wrt U
        - Returns:
            - reshaped_grad: (T,(x_dim+u_dim),T,u_dim) tensor wrt traj
        - NOTE: current implementation assumes trajectory wrt U is linear hence we do not
            update traj grads wrt to U
        """
        # traj_grads -> (T,(x_dim+u_dim),T,u_dim) wrt U
        x_dim = self.p_dim - self.dim
        incoming_grad = torch.cat(
            (torch.zeros(self.horizon, self.p_dim, self.horizon, x_dim, **self.tensor_kwargs),
                incoming_grad),
                dim=-1)
        # traj_grad -> (T,(x_dim+u_dim),T,(x_dim+u_dim)) wrt traj
        incoming_grad = incoming_grad.view(self.horizon * self.p_dim, self.horizon * self.p_dim)
        # traj_grad -> (T*(x_dim+u_dim),T*(x_dim+u_dim)) wrt traj
        return incoming_grad

    def set_traj_grad_s(self, traj_grads_U_s) -> None:
        """
        - Inputs:
            - traj_grads_U_s: (T,(x_dim+u_dim),T,u_dim) tensor wrt U
        """
        if traj_grads_U_s is not None:
            self.traj_grads_s = self._propagate_incoming_grad(traj_grads_U_s)

    def set_traj_grad_t(self, traj_grads_U_t) -> None:
        """
        - Inputs:
            - traj_grads_U_t: (T,(x_dim+u_dim),T,u_dim) tensor wrt U
        """
        if traj_grads_U_t is not None:
            self.traj_grads_t = self._propagate_incoming_grad(traj_grads_U_t)

    def _log_likelihood(self, x_s, x_t) -> torch.Tensor:
        """
        - Inputs:
            - x_s: (...,Ns,T*x_dim*3) tensor
            - x_t: (...,Nt,T*x_dim*3) tensor
        - Returns:
            - cost: (...,Ns,Nt) tensor
        """
        batch_shape = x_s.shape[:-1]
        x_s = x_s.view(*batch_shape, self.horizon, self.p_dim)
        x_t = x_t.view(*batch_shape, self.horizon, self.p_dim)
        return self.cross_collision_cost_fn.cost(x_s, x_t)

    def _grad_log_likelihood(self, x_s, x_t) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - Inputs:
            - x_s: (...,Ns,T*dim*3) tensor
            - x_t: (...,Nt,T*dim*3) tensor
        - Returns:
            - grad_cost_s: (...,Ns,Nt,T*dim*3) tensor
            - grad_cost_t: (...,Ns,Nt,T*dim*3) tensor
            - cost: (...,Ns,Nt) tensor
        """
        # NOTE: current implementation assumes trajectory wrt U is linear hence we do not update traj grads wrt to U
        batch_shape = x_s.shape[:-2]
        Ns = x_s.shape[-2]
        Nt = x_t.shape[-2]

        x_s = x_s.view(*batch_shape, Ns, self.horizon, self.p_dim)
        x_t = x_t.view(*batch_shape, Nt, self.horizon, self.p_dim)

        traj_s_cost_grad, traj_t_cost_grad, log_px = self.cross_collision_cost_fn.grad_w_cost(x_s, x_t)
        # dc/d(U_s) = dc/d(X_bar_s) @ d(X_bar_s)/d(U_s)
        grad_log_px_s = traj_s_cost_grad.view(*batch_shape, Ns, Nt, 1, self.horizon * (self.p_dim)) \
            @ self.traj_grads_s
        grad_log_px_s = grad_log_px_s.view(*batch_shape, Ns, Nt, self.horizon * self.p_dim)
        # dc/d(U_t) = dc/d(X_bar_t) @ d(X_bar_t)/d(U_t)
        grad_log_px_t = traj_t_cost_grad.view(*batch_shape, Ns, Nt, 1, self.horizon * (self.p_dim)) \
            @ self.traj_grads_t
        grad_log_px_t = grad_log_px_t.view(*batch_shape, Ns, Nt, self.horizon * self.p_dim)

        return grad_log_px_s, grad_log_px_t, log_px
