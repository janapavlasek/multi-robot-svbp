"""
Implements linearized gaussian factor forms for trajectory factors by setting those factors as h
- NOTE: Actual factor used is the linearized gaussian energy function, which linearizes h about a given linearization pt
"""

from typing import Tuple, Union
import torch
from multi_robot_svbp.sim.robot import DynamicsModel
from multi_robot_svbp.factors.linear_gaussian_base_factors import LinearGaussianUnaryRobotFactor, LinearGaussianPairwiseRobotFactor
from multi_robot_svbp.costs.trajectory_costs import RunningDeltaCost, RunningDeltaJacCost, RunningCollisionCost
from multi_robot_svbp.costs.obstacle_costs import SignedDistanceMap2DCost, ExponentialSumObstacleCost, KBendingObstacleCost


class LinearGaussianRunningCostFactor(LinearGaussianUnaryRobotFactor):
    """
    Linearized Gaussian Factor implementation of Running Cost.
    - This is split off from UnaryRobotTrajectoryFactor to allow the bias of each component factors to be controlled
        seperately.
    - Also it is advised not to cost against goal position since we cannot get a good estimate on what the deviation
        of the pose from the goal should be like since it changes depending on the robot's starting location
    """
    def __init__(self,
                 U_linearization_pt: torch.Tensor, x_0_linearization_pt: torch.Tensor,
                 bias: torch.Tensor, bias_sigma: torch.Tensor,
                 model: Union[None,DynamicsModel]=None,
                 c_pos=0., c_vel=0.25, c_u=0.2,
                 dim=2, horizon=1, goal=None,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float}):
        """
        - Inputs:
            - U_linearization_pt: (T,dim) tensor
            - x_0_linearization_pt: (dim,) tensor
            - bias: (1,) tensor
            - bias_sigma: (1,1) tensor
            - model: None | DynamicsModel, robot dynamics model, None if not known now
            - c_pos: float, scalar cost for distance from running target pose
            - c_vel: float, scalar cost for distance from running target vel
            - c_u: float, scalar cost for distance from running target u
            - dim: int, dimension of workspace
            - horizon: int, length of projection of rollouts
            - goal: (dim,) tensor, goal position to reach for this agent
            - tensor_kwargs: dict, keyword args used for creating new tensors
        """
        self._dim = dim
        self._T = horizon
        self.U_linearization_pt = U_linearization_pt
        self._x_0_linearization_pt = x_0_linearization_pt
        running_cost_Qs = (c_pos * torch.eye(dim), c_vel * torch.eye(dim), c_u * torch.eye(dim))
        running_cost_x_bars = None if goal is None else (goal, torch.zeros_like(goal), torch.zeros_like(goal))
        self.running_cost_fn = RunningDeltaCost(Qs=running_cost_Qs, x_bars=running_cost_x_bars,
                                                tensor_kwargs=tensor_kwargs)
        super().__init__(self.running_cost_fn.grad_w_cost,
                         bias, bias_sigma,
                         U_linearization_pt, x_0_linearization_pt,
                         False,
                         model)

    def reshape_traj_grads_wrt_U(self, traj_grads_U: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping trajectory gradients to use for gradient propagation
        - Inputs:
            - traj_grads_U: (T,dim*3,T,dim) tensor wrt U
        - Returns:
            - traj_grads_U: (T*dim*3,T*dim) tensor wrt U
        """
        return traj_grads_U.view(self._T*self._dim*3, self._T*self._dim)

    def reshape_h_grads_wrt_traj(self, h_grads_traj: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping h(traj) gradients to use for gradient propagation
        - Inputs:
            - h_grads_traj: (T,dim*3) tensor wrt U
        - Returns:
            - h_grads_traj: (1,T*dim*3)
        """
        return h_grads_traj.view(1, self._T*self._dim*3)

    def reshape_traj_for_fn(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Reshape trajectory from model for use by h(x)
        - Inputs:
            - traj: (T,dim*3) tensor
        - Outpus:
            - traj: (T,dim3) tensor (no change)
        """
        return traj

    def reshape_U_for_model(self, U: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping U for model
        - Inputs:
            - U: (T*dim) tensor
        - Inputs:
            - U: (T,dim) tensor
        """
        return U.view(self._T, self._dim)


class LinearGaussianRunningCostJacFactor(LinearGaussianUnaryRobotFactor):
    """
    Linearized Gaussian Factor implementation of Running Cost.
    - Instead of using the quadratic cost, use the jacobian to find the minimum solution by finding jac_h(x)=0
        instead of trying to set h(x) to 0
    """
    def __init__(self,
                 U_linearization_pt: torch.Tensor, x_0_linearization_pt: torch.Tensor,
                 bias: torch.Tensor, bias_sigma: torch.Tensor,
                 model: Union[None,DynamicsModel]=None,
                 c_pos=0., c_vel=0.25, c_u=0.2,
                 dim=2, horizon=1, goal=None,
                 alpha=1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float}):
        """
        - Inputs:
            - U_linearization_pt: (T,dim) tensor
            - x_0_linearization_pt: (dim,) tensor
            - bias: (1,) tensor
            - bias_sigma: (1,1) tensor
            - model: None | DynamicsModel, robot dynamics model, None if not known now
            - c_pos: float, scalar cost for distance from running target pose
            - c_vel: float, scalar cost for distance from running target vel
            - c_u: float, scalar cost for distance from running target u
            - dim: int, dimension of workspace
            - horizon: int, length of projection of rollouts
            - goal: (dim,) tensor, goal position to reach for this agent
            - tensor_kwargs: dict, keyword args used for creating new tensors
        """
        self._dim = dim
        self._T = horizon
        self.U_linearization_pt = U_linearization_pt
        self._x_0_linearization_pt = x_0_linearization_pt
        running_cost_Qs = (c_pos * torch.eye(dim), c_vel * torch.eye(dim), c_u * torch.eye(dim))
        running_cost_x_bars = None if goal is None else (goal, torch.zeros_like(goal), torch.zeros_like(goal))
        self.running_cost_jac = RunningDeltaJacCost(Qs=running_cost_Qs, x_bars=running_cost_x_bars,
                                                    tensor_kwargs=tensor_kwargs)
        super().__init__(self.running_cost_jac.grad_w_cost,
                         bias, bias_sigma,
                         U_linearization_pt, x_0_linearization_pt,
                         False,
                         model,
                         alpha)

    def set_model(self, model: DynamicsModel) -> None:
        """
        Only allow linear models, since the jac of linear models is a constant which become a scaling factor
        when propagating gradient. (Remember that this factor is a jacobian of h(R(U,x_0)))
        """
        assert model.is_linear(), f"For now only accept linear models"
        return super().set_model(model)

    def reshape_traj_grads_wrt_U(self, traj_grads_U: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping trajectory gradients to use for gradient propagation
        - Inputs:
            - traj_grads_U: (T,dim*3,T,dim) tensor wrt U
        - Returns:
            - traj_grads_U: (T*dim*3,T*dim) tensor wrt U
        """
        return traj_grads_U.view(self._T*self._dim*3, self._T*self._dim)

    def reshape_h_grads_wrt_traj(self, h_grads_traj: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping h(traj) gradients to use for gradient propagation
        - Inputs:
            - h_grads_traj: (T,dim*3,T,dim*3) tensor wrt X
        - Returns:
            - h_grads_traj: (T*dim*3,T*dim*3)
        """
        return h_grads_traj.view(self._T*self._dim*3, self._T*self._dim*3)

    def reshape_traj_for_fn(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Reshape trajectory from model for use by h(x)
        - Inputs:
            - traj: (T,dim*3) tensor
        - Outpus:
            - traj: (T,dim3) tensor (no change)
        """
        return traj

    def reshape_U_for_model(self, U: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping U for model
        - Inputs:
            - U: (T*dim) tensor
        - Inputs:
            - U: (T,dim) tensor
        """
        return U.view(self._T, self._dim)


class LinearGaussianExponentialSumObstacleFactor(LinearGaussianUnaryRobotFactor):
    """
    Linearized Gaussian Factor implementation of ExponentialSumObstacleCost
    - This is split off from UnaryRobotTrajectoryFactor to allow the bias of each component factors to be controlled
        seperately.
    """
    def __init__(self,
                 U_linearization_pt: torch.Tensor, x_0_linearization_pt: torch.Tensor,
                 bias: torch.Tensor, bias_sigma: torch.Tensor,
                 model: Union[None,DynamicsModel]=None,
                 c_obs=10000, sigma_obs_out=0, sigma_obs_in=50,
                 dim=2, horizon=1,
                 signed_dist_map_fn : SignedDistanceMap2DCost = None,
                 alpha=1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float}):
        """
        - Inputs:
            - U_linearization_pt: (T,dim) tensor
            - x_0_linearization_pt: (dim,) tensor
            - bias: (1,) tensor
            - bias_sigma: (1,1) tensor
            - model: None | DynamicsModel, robot dynamics model, None if not known now
            - sigma_obs_out : float, scaling factor when closer to an obstacle
            - sigma_obs_in : float, scaling factor when breaching an obstacle
            - sigma_obs: float, scalar multiplier for increasing cost when higher obstacle violation is detected
            - dim: int, dimension of workspace
            - horizon: int, length of projection of rollouts
            - goal: (dim,) tensor, goal position to reach for this agent
            - signed_dist_map_fn: SignedDistanceMap2DCost, map used to calculate collision with environment
            - tensor_kwargs: dict, keyword args used for creating new tensors
        """
        self._dim = dim
        self._T = horizon
        self.U_linearization_pt = U_linearization_pt
        self._x_0_linearization_pt = x_0_linearization_pt
        self.obs_cost_fn = ExponentialSumObstacleCost(signed_2d_map=signed_dist_map_fn,
                                                      sigma_obs_out=sigma_obs_out, sigma_obs_in=sigma_obs_in, sigma=c_obs,
                                                      tensor_kwargs=tensor_kwargs)
        super().__init__(self.obs_cost_fn.grad_w_cost,
                         bias, bias_sigma,
                         U_linearization_pt, x_0_linearization_pt,
                         False,
                         model,
                         alpha)

    def reshape_traj_grads_wrt_U(self, traj_grads_U: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping trajectory gradients to use for gradient propagation
        - Inputs:
            - traj_grads_U: (T,dim*3,T,dim) tensor wrt U
        - Returns:
            - traj_grads_U: (T*dim*3,T*dim) tensor wrt U
        """
        return traj_grads_U.view(self._T*self._dim*3, self._T*self._dim)

    def reshape_h_grads_wrt_traj(self, h_grads_traj: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping h(traj) gradients to use for gradient propagation
        - Inputs:
            - h_grads_traj: (T,dim*3) tensor wrt X
        - Returns:
            - h_grads_traj: (1,T*dim*3)
        """
        return h_grads_traj.view(1, self._T*self._dim*3)

    def reshape_traj_for_fn(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Reshape trajectory from model for use by h(x)
        - Inputs:
            - traj: (T,dim*3) tensor
        - Outpus:
            - traj: (T,dim*3) tensor (no change)
        """
        return traj

    def reshape_U_for_model(self, U: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping U for model
        - Inputs:
            - U: (T*dim) tensor
        - Inputs:
            - U: (T,dim) tensor
        """
        return U.view(self._T, self._dim)


class LinearGaussianKBendingObstacleFactor(LinearGaussianUnaryRobotFactor):
    """
    Linearized Gaussian Factor implementation of KBendingObstacleCost
    - New obstacle cost formulation that does not sum individual components and does not have an exponential term
    """
    def __init__(self,
                 U_linearization_pt: torch.Tensor, x_0_linearization_pt: torch.Tensor,
                 bias: torch.Tensor, bias_sigma: torch.Tensor,
                 model: Union[None,DynamicsModel]=None,
                 critical_dist= .5, k_bend= .3, sigma_T: Union[torch.Tensor, float]= 1.,
                 dim=2, horizon=1,
                 signed_dist_map_fn : SignedDistanceMap2DCost = None,
                 alpha=1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float}):
        """
        - Inputs:
            - U_linearization_pt: (T,dim) tensor
            - x_0_linearization_pt: (dim,) tensor
            - bias: (1,) tensor
            - bias_sigma: (1,1) tensor
            - model: None | DynamicsModel, robot dynamics model, None if not known now
            - critial_dist : float, distance we expect the point to be away from obstacles
            - k_bend : float, bending factor that bends the cost function to be
                linear (k_bend=1), convex (k_bend=(0,1]) or concave (k_bend=[1~inf))
            - sigma_T : (T,) tensor or float, if tensor is given, scales each time component seperately and
                input will be fixed to horizons of size T
            - dim: int, dimension of workspace
            - horizon: int, length of projection of rollouts
            - goal: (dim,) tensor, goal position to reach for this agent
            - signed_dist_map_fn: SignedDistanceMap2DCost, map used to calculate collision with environment
            - tensor_kwargs: dict, keyword args used for creating new tensors
        """
        self._dim = dim
        self._T = horizon
        self.U_linearization_pt = U_linearization_pt
        self._x_0_linearization_pt = x_0_linearization_pt
        self.obs_cost_fn = KBendingObstacleCost(signed_2d_map=signed_dist_map_fn,
                                                radius=critical_dist, k_bend=k_bend,
                                                sigma_T=sigma_T,
                                                tensor_kwargs=tensor_kwargs)
        super().__init__(self.obs_cost_fn.grad_w_cost,
                         bias, bias_sigma,
                         U_linearization_pt, x_0_linearization_pt,
                         False,
                         model,
                         alpha)

    def reshape_traj_grads_wrt_U(self, traj_grads_U: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping trajectory gradients to use for gradient propagation
        - Inputs:
            - traj_grads_U: (T,dim*3,T,dim) tensor wrt U
        - Returns:
            - traj_grads_U: (T*dim*3,T*dim) tensor wrt U
        """
        return traj_grads_U.view(self._T*self._dim*3, self._T*self._dim)

    def reshape_h_grads_wrt_traj(self, h_grads_traj: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping h(traj) gradients to use for gradient propagation
        - Inputs:
            - h_grads_traj: (T,T,dim*3) tensor wrt X
        - Returns:
            - h_grads_traj: (T,T*dim*3)
        """
        return h_grads_traj.view(self._T, self._T*self._dim*3)

    def reshape_traj_for_fn(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Reshape trajectory from model for use by h(x)
        - Inputs:
            - traj: (T,dim*3) tensor
        - Outpus:
            - traj: (T,dim*3) tensor (no change)
        """
        return traj

    def reshape_U_for_model(self, U: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping U for model
        - Inputs:
            - U: (T*dim) tensor
        - Inputs:
            - U: (T,dim) tensor
        """
        return U.view(self._T, self._dim)


class LinearGaussianTrajectoryCollisionFactor(LinearGaussianPairwiseRobotFactor):
    """
    Linearized Gaussian Factor implementation of TrajectoryCollisionFactor

    NOTE: initial positions (x_0) are important as an argument, but is treated as a constant we cannot
    change (since current position is determined by the current physical locality of the robot at a given
    time) -> this may not be true for other use cases such as localization etc (where position is a variable
    to be determined)
    """
    def __init__(self,
                 U_s_linearization_pt: torch.Tensor, x_0_s_linearization_pt: torch.Tensor,
                 U_t_linearization_pt: torch.Tensor, x_0_t_linearization_pt: torch.Tensor,
                 bias: torch.Tensor, bias_sigma: torch.Tensor,
                 model_s: Union[None,DynamicsModel]=None, model_t: Union[None,DynamicsModel]=None,
                 c_coll=100., c_coll_end=None,
                 horizon=1, dim=2, r=0.5, k=0.3,
                 alpha=1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float}, **kwargs):
        """
        - Inputs:
            - U_s_linearization_pt: (T,dim) tensor
            - x_0_s_linearization_pt: (dim,) tensor
            - U_t_linearization_pt: (T,dim) tensor
            - x_0_t_linearization_pt: (dim,) tensor
            - bias: (1,) tensor
            - bias_sigma: (1,1) tensor
            - model_s: None | DynamicsModel, robot s dynamics model, None if not known now
            - model_t: None | DynamicsModel, robot t dynamics model, None if not known now
            - c_coll: float, scalar cost for colliding with another robot
            - c_coll_end: None | float, if defined, changes c_coll to scale accross rollout linearly till furthest
                rollout projection c_coll value equals to c_coll_end
            - dim: int, dimension of workspace
            - horizon: int, length of projection of rollouts
            - r: float, radius of agents below which they are in collision
            - k: float, linearity bending factor, controls the curve of the function
                1: linear, >1: convex, <1: concave
            - tensor_kwargs: dict, keyword args used for creating new tensors
            - kwargs: keyword args, to pass to TrajectoryCollisionFactor factor
        """
        self.dim = dim
        self.T = horizon
        sigma_T = c_coll if c_coll_end is None else torch.linspace(c_coll, c_coll_end, horizon, **tensor_kwargs)
        self.running_collision_cost = RunningCollisionCost(
            pos_dim=dim, radius=r, k_bend=k, sigma_T=sigma_T, tensor_kwargs=tensor_kwargs)
        self.tensor_kwargs = tensor_kwargs
        super().__init__(self.running_collision_cost.grad_w_cost,
                         bias, bias_sigma,
                         U_s_linearization_pt, x_0_s_linearization_pt,
                         U_t_linearization_pt, x_0_t_linearization_pt,
                         False,
                         model_s, model_t,
                         alpha)

    def reshape_traj_grads_wrt_U(self, traj_s_grads_U: torch.Tensor, traj_t_grads_U: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for reshaping trajectory gradients to use for gradient propagation
        - Inputs:
            - traj_s_grads_U: (T,dim*3,T,dim) tensor wrt U from model
            - traj_s_grads_U: (T,dim*3,T,dim) tensor wrt U from model
        - Returns:
            - traj_s_grads_U: (T*dim*3,T*dim) tensor wrt U for propagation
            - traj_s_grads_U: (T*dim*3,T*dim) tensor wrt U for propagation
        """
        return traj_s_grads_U.view(self.T*self.dim*3, self.T*self.dim), \
            traj_t_grads_U.view(self.T*self.dim*3, self.T*self.dim)

    def reshape_h_grads_wrt_traj(self, h_grads_traj_s: torch.Tensor, h_grads_traj_t: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for reshaping h(traj) gradients to use for gradient propagation
        - Inputs:
            - h_grads_traj_s: (T,dim*3) tensor wrt traj_s
            - h_grads_traj_t: (T,dim*3) tensor wrt traj_t
        - Returns:
            - h_grads_traj_s: (1,T*dim*3) tensor wrt traj_s
            - h_grads_traj_t: (1,T*dim*3) tensor wrt traj_t
        """
        return h_grads_traj_s.view(1, self.T*self.dim*3), h_grads_traj_t.view(1, self.T*self.dim*3)

    def reshape_U_for_models(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for reshaping U for model_s and model_t
        - Inputs:
            - U: (T*dim+T*dim) tensor
        - Inputs:
            - U_s: (T,dim) tensor to model_s
            - U_t: (T,dim) tensor to model_t
        """
        U_s = U[:(self.T*self.dim)].view(self.T, self.dim)
        U_t = U[(self.T*self.dim):].view(self.T, self.dim)
        return U_s, U_t

    def reshape_trajs_for_fn(self, traj_s: torch.Tensor, traj_t: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape trajectory from model for use by h(x)
        - Inputs:
            - traj_s: (T,dim*3) tensor from model_s
            - traj_t: (T,dim*3) tensor from model_t
        - Outpus:
            - traj_s: (T,dim*3) tensor to h(traj_s, traj_t) (No change)
            - traj_t: (T,dim*3) tensor to h(traj_s, traj_t) (No change)
        """
        return traj_s, traj_t