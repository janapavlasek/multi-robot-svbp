"""
Implements generic forms of linearized gaussian factor
- NOTE: Actual factor used is the linearized gaussian energy function, which linearizes h about a given linearization pt
"""

from typing import Callable, Tuple, Union
import torch
from torch_bp.graph.factors import UnaryFactor, PairwiseFactor
from torch_bp.graph.factors.linear_gaussian_factors import UnaryGaussianLinearFactor, PairwiseGaussianLinearFactor
from multi_robot_svbp.sim.robot import DynamicsModel

class DelayedUnaryGaussianLinearFactor(UnaryFactor):
    """
    Functionally same as UnaryGaussianLinearFactor, but with the ability to delay the definition of grad_w_h
    so that the dynamics model could be passed in at a later time
    """
    def __init__(self, grads_w_h : Union[None, Callable],
                 z : torch.Tensor, sigma : torch.Tensor,
                 x_0 : torch.Tensor,
                 h_linear=True,
                 alpha=1) -> None:
        """
        Inputs:
        - h : None | Callable, function which evaluates h(x), None if delay creation of factor
        - h_w_grads : Callable, function which evaluates (jac_h(x), h(x))
        - z : (x_out_dim,) tensor, bias or the mean we want h(x) to have
        - sigma : (x_out_dim, x_out_dim) tensor, covariance we want h(x) to have
        - h_linear : bool, whether h(x) is linear
        - alpha : float, scaling of factor
        """
        super().__init__(alpha)
        self.z = z
        self.sigma = sigma
        self.x_0 = x_0
        self.alpha = alpha
        self.factor = None
        if grads_w_h is not None:
            self.factor = UnaryGaussianLinearFactor(grads_w_h, z, sigma, x_0, h_linear, alpha)

    def set_grad_w_h(self, grads_w_h: Callable, linear: bool):
        """
        Instantiate actual linear factor now instead on upon creation of this class
        - Inputs:
            - grads_w_h: Callable, function which evaluates h(x)
            - linear: bool, whether h(x) is linear
        """
        self.factor = UnaryGaussianLinearFactor(grads_w_h, self.z, self.sigma, self.x_0, linear, self.alpha)

    def log_likelihood(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
        - x : (eta tensor, lambda tensor)
        Returns:
        - eta : (x_dim,) tensor
        - lambda : (x_dim,x_dim) tensor
        """
        self.x_0 = x
        return self.factor(x)

    def update_bias(self, new_z: torch.Tensor, new_sigma: Union[None, torch.Tensor]= None):
        """
        Updates the bias and optionally the sigma
        """
        self.factor.update_bias(new_z, new_sigma)


class DelayedPairwiseGaussianLinearFactor(PairwiseFactor):
    """
    Functionally same as PairwiseGaussianLinearFactor, but with the ability to delay the definition of grad_w_h
    so that the dynamics model could be passed in at a later time
    """
    def __init__(self, grads_w_h : Union[None, Callable],
                 z : torch.Tensor, sigma : torch.Tensor,
                 x_0 : torch.Tensor,
                 h_linear : bool,
                 alpha=1) -> None:
        """
        Inputs:
        - h : None | Callable, function which evaluates h(x), None if delay creation of factor
        - h_w_grads : Callable, function which evaluates (jac_h(x), h(x))
        - z : (x_out_dim,) tensor, bias or the mean we want h(x) to have
        - sigma : (x_out_dim, x_out_dim) tensor, covariance we want h(x) to have
        - linear : bool, whether h(x) is linear
        - alpha : float, scaling of factor
        """
        super().__init__(alpha)
        self.z = z
        self.sigma = sigma
        self.x_0 = x_0
        self.alpha = alpha
        self.factor = None
        if grads_w_h is not None:
            self.factor = PairwiseGaussianLinearFactor(grads_w_h, z, sigma, x_0, h_linear, alpha)

    def set_grad_w_h(self, grads_w_h: Callable, linear: bool):
        """
        Instantiate actual linear factor now instead on upon creation of this class
        - Inputs:
            - grads_w_h: Callable, function which evaluates h(x)
            - linear: bool, whether h(x) is linear
        """
        self.factor = PairwiseGaussianLinearFactor(grads_w_h, self.z, self.sigma, self.x_0, linear, self.alpha)

    def log_likelihood(self, x_s, x_t):
        """
        Inputs:
        - x_s : (eta tensor, lambda tensor)
        - x_t : (eta tensor, lambda tensor)
        Returns:
        - eta : (x_dim,) tensor
        - lambda : (x_dim,x_dim) tensor
        """
        return self.factor(x_s, x_t)

    def update_bias(self, new_z: torch.Tensor, new_sigma: Union[None, torch.Tensor]= None):
        """
        Updates the bias and optionally the sigma
        """
        self.factor.update_bias(new_z, new_sigma)


class LinearGaussianUnaryRobotFactor(DelayedUnaryGaussianLinearFactor):
    """
    Generic Linear Gaussian Unary Factor which is dependent on the robot dynamics model.
    - For systems we want to infer the controls but work with functions that acts on the trajectory instead of the
        controls.
    - Has the ability to delay the instantiation of the actual factor until the model is known. Useful when
        responsibility of defining the model is seperate from the generation of the factor
    - NOTE: initial position (x_0) is important as an argument, but is treated as a constant we cannot
        change (since current position is determined by the current physical locality of the robot at a given
        time) -> this may not be true for other use cases such as localization etc (where position is a variable
        to be determined)
    """
    def __init__(self, grads_w_h: Union[None, Callable],
                 z: torch.Tensor, sigma: torch.Tensor,
                 U_linearization_pt: torch.Tensor, x_0_linearization_pt: torch.Tensor,
                 h_linear=True,
                 model: Union[None,DynamicsModel]=None,
                 alpha=1) -> None:
        self.U_linearization_pt = U_linearization_pt
        self._x_0_linearization_pt = x_0_linearization_pt
        self._grads_w_h_traj = grads_w_h
        self._full_grads_w_h = None
        self.h_linear = h_linear
        linear = h_linear
        if model is not None:
            self._model = model
            if model.is_linear():
                traj_grad_U, _, _ = model.rollout_w_grad(self.U_linearization_pt, self._x_0_linearization_pt)
                self._traj_grad_U = self.reshape_traj_grads_wrt_U(traj_grad_U)
                self._full_grads_w_h = self._full_grad_w_h_linear_model
            else:
                self._full_grads_w_h = self._full_grad_w_h_nonlinear_model
                linear = False
        super().__init__(self._full_grads_w_h, z, sigma, U_linearization_pt.view(-1), linear, alpha)

    def set_model(self, model: DynamicsModel) -> None:
        """
        Used to set the model at a later time
        - Inputs:
            - model: Dynamics model
        """
        self._model = model
        if model.is_linear():
            traj_grad_U, _, _ = model.rollout_w_grad(self.U_linearization_pt, self._x_0_linearization_pt)
            self._traj_grad_U = self.reshape_traj_grads_wrt_U(traj_grad_U)
            self._full_grads_w_h = self._full_grad_w_h_linear_model
        else:
            self._full_grads_w_h = self._full_grad_w_h_nonlinear_model
        linear = self.h_linear and model.is_linear()
        self.set_grad_w_h(self._full_grads_w_h, linear)

    def set_x_0(self, x_0: torch.Tensor) -> None:
        """
        We treat x_0 as a constant and not a inferred variable
        - Inputs:
            - x_0: (x_dim,) tensor -> shape to be passed to model
        """
        self._x_0_linearization_pt = x_0

    def _full_grad_w_h_linear_model(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If model is linear, no need to update model jacobians
        - Inputs:
            - U: (T*dim) tensor, control inputs
        - Returns:
            - grad_cost: (1,T*dim) tensor, wrt U
            - cost: (1,) tensor, scalar cost
        """
        traj = self._model.rollout(self.reshape_U_for_model(U), self._x_0_linearization_pt)
        traj = self.reshape_traj_for_fn(traj)

        grad_h_traj, h_traj = self._grads_w_h_traj(traj)
        grad_h_U = self.reshape_h_grads_wrt_traj(grad_h_traj) @ self._traj_grad_U

        return grad_h_U, h_traj.view(-1)

    def _full_grad_w_h_nonlinear_model(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If model is nonlinear, find jacobians all the time
        - Inputs:
            - U: (T*dim) tensor, control inputs
        - Returns:
            - grad_cost: (1,T*dim) tensor, wrt U
            - cost: (1,) tensor, scalar cost
        """
        traj_grad_U, traj = self._model.rollout_w_grad(self.reshape_U_for_model(U), self._x_0_linearization_pt)
        traj_grad_U = self.reshape_traj_grads_wrt_U(traj_grad_U)
        traj = self.reshape_traj_for_fn(traj)

        grad_h_traj, h_traj = self.running_cost_fn.grad_w_cost(traj)
        grad_h_U = self.reshape_h_grads_wrt_traj(grad_h_traj) @ traj_grad_U

        return grad_h_U, h_traj.view(-1)

    def reshape_traj_grads_wrt_U(self, traj_grads_U: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping trajectory gradients to use for gradient propagation
        - Inputs:
            - traj_grads_U: tensor wrt U
        - Returns:
            - traj_grads_U: tensor wrt U
        """
        raise NotImplementedError

    def reshape_h_grads_wrt_traj(self, h_grads_traj: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping h(traj) gradients to use for gradient propagation
        - Inputs:
            - h_grads_traj: tensor wrt X
        - Returns:
            - h_grads_traj: tensor wrt X
        """
        raise NotImplementedError

    def reshape_traj_for_fn(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Reshape trajectory from model for use by h(x)
        - Inputs:
            - traj: tensor
        - Outpus:
            - traj: tensor
        """
        raise NotImplementedError

    def reshape_U_for_model(self, U: torch.Tensor) -> torch.Tensor:
        """
        Method for reshaping U for model
        - Inputs:
            - U: tensor
        - Inputs:
            - U: tensor
        """
        raise NotImplementedError


class LinearGaussianPairwiseRobotFactor(DelayedPairwiseGaussianLinearFactor):
    """
    Generic Linear Gaussian Pairwise Factor which is dependent on the robot dynamics model.
    - For systems we want to infer the controls but work with functions that acts on the trajectory instead of the
        controls.
    - Has the ability to delay the instantiation of the actual factor until the model is known. Useful when
        responsibility of defining the model is seperate from the generation of the factor
    - NOTE: initial position (x_0) is important as an argument, but is treated as a constant we cannot
        change (since current position is determined by the current physical locality of the robot at a given
        time) -> this may not be true for other use cases such as localization etc (where position is a variable
        to be determined)
    """
    def __init__(self, grads_w_h: Union[None, Callable],
                 z: torch.Tensor, sigma: torch.Tensor,
                 U_s_linearization_pt: torch.Tensor, x_0_s_linearization_pt: torch.Tensor,
                 U_t_linearization_pt: torch.Tensor, x_0_t_linearization_pt: torch.Tensor,
                 h_linear: bool,
                 model_s: Union[None,DynamicsModel]=None, model_t: Union[None,DynamicsModel]=None,
                 alpha=1) -> None:
        self.U_s_linearization_pt = U_s_linearization_pt
        self.U_t_linearization_pt = U_t_linearization_pt
        self.x_0_s_linearization_pt = x_0_s_linearization_pt
        self.x_0_t_linearization_pt = x_0_t_linearization_pt
        self.model_s = model_s
        self.model_t = model_t
        self.h_linear = h_linear
        self.traj_grad_U_s, self.traj_grad_U_t = None, None
        self._grad_w_trajectory_s, self._grad_w_trajectory_t = None, None
        linear = h_linear
        if model_s is not None:
            self.traj_grad_U_s, _, _ = model_s.rollout_w_grad(U_s_linearization_pt, x_0_s_linearization_pt)
            self._grad_w_trajectory_s = self._grad_w_trajectory_linear if model_s.is_linear() \
                else self._grad_w_trajectory_nonlinear
            linear &= model_s.is_linear()
        if model_t is not None:
            self.traj_grad_U_t, _, _ = model_t.rollout_w_grad(U_t_linearization_pt, x_0_t_linearization_pt)
            self._grad_w_trajectory_t = self._grad_w_trajectory_linear if model_t.is_linear() \
                else self._grad_w_trajectory_nonlinear
            linear &= model_t.is_linear()
        self._grads_w_h = grads_w_h
        grads_w_h = self._full_grads_w_h if (model_s is not None) and (model_t is not None) else None
        U_linearization_pt = torch.cat((U_s_linearization_pt.view(-1), U_t_linearization_pt.view(-1)))
        super().__init__(grads_w_h, z, sigma, U_linearization_pt, linear, alpha)

    def set_model(self, model_s: DynamicsModel, model_t: DynamicsModel) -> None:
        """
        Used to set the model at a later time
        - Inputs:
            - model_s: Dynamics model for node s
            - model_t: Dynamics model for node t
        """
        self.model_s = model_s
        self.traj_grad_U_s, _, _ = model_s.rollout_w_grad(self.U_s_linearization_pt, self.x_0_s_linearization_pt)
        self._grad_w_trajectory_s = self._grad_w_trajectory_linear if model_s.is_linear() \
            else self._grad_w_trajectory_nonlinear
        self.model_t = model_t
        self.traj_grad_U_t, _, _ = model_t.rollout_w_grad(self.U_t_linearization_pt, self.x_0_t_linearization_pt)
        self._grad_w_trajectory_t = self._grad_w_trajectory_linear if model_t.is_linear() \
            else self._grad_w_trajectory_nonlinear
        self.set_grad_w_h(self._full_grads_w_h, self.h_linear and self.model_s.is_linear() and self.model_t.is_linear())

    def set_x_0(self, x_0_s: torch.Tensor, x_0_t: torch.Tensor) -> None:
        """
        We treat x_0 as a constant and not a inferred variable
        - Inputs:
            - x_0_s: (x_dim,) tensor
            - x_0_t: (x_dim,) tensor
        """
        self.x_0_s_linearization_pt = x_0_s
        self.x_0_t_linearization_pt = x_0_t

    def _grad_w_trajectory_linear(self, U: torch.Tensor, x_0: torch.Tensor, model: DynamicsModel,
                                  prev_traj_grad_U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Binded function to get new trajectory from model, as well as update internal grad values if required
        - Inputs:
            - U: tensor, control inputs
            - x_0: tensor, initial pose
            - model: DynamicModel, model linked to this trajectory update
            - prev_traj_grad_U: tensor, previous grad u
        - Returns:
            - grad_U: tensor, trajectory jacobian wrt U
            - traj: tensor, trajectory
        """
        return prev_traj_grad_U, model.rollout(U, x_0)

    def _grad_w_trajectory_nonlinear(self, U: torch.Tensor, x_0: torch.Tensor, model: DynamicsModel,
                                     prev_traj_grad_U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Binded function to get new trajectory from model, as well as update internal grad values if required
        - Inputs:
            - U: tensor, control inputs
            - x_0: tensor, initial pose
            - model: DynamicModel, model linked to this trajectory update
            - prev_traj_grad_U: tensor, previous grad u
        - Returns:
            - grad_U: tensor, trajectory jacobian wrt U
            - traj: tensor, trajectory
        """
        return model.rollout_w_grad(U, x_0)

    def _full_grads_w_h(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Inputs:
            - U: tensor, concatonated control inputs from node s and t (ie [U_s, U_t])
        - Returns:
            - grad_cost_U_s: tensor, grad of cost wrt U_s
            - grad_cost_U_t: tensor, grad of cost wrt U_t
            - cost: tensor, factor cost
        """
        U_s, U_t = self.reshape_U_for_models(U)
        self.U_s_linearization_pt = U_s
        self.U_s_linearization_pt = U_t

        traj_grad_U_s, traj_s = self._grad_w_trajectory_s(U_s, self.x_0_s_linearization_pt, self.model_s,
                                                          self.traj_grad_U_s)
        traj_grad_U_t, traj_t = self._grad_w_trajectory_s(U_t, self.x_0_t_linearization_pt, self.model_t,
                                                          self.traj_grad_U_t)
        traj_grad_U_s, traj_grad_U_t = self.reshape_traj_grads_wrt_U(traj_grad_U_s, traj_grad_U_t)

        cost_grad_traj_s, cost_grad_traj_t, cost = self._grads_w_h(traj_s, traj_t)
        cost_grad_traj_s, cost_grad_traj_t = self.reshape_h_grads_wrt_traj(cost_grad_traj_s, cost_grad_traj_t)
        cost_grad_U_s = cost_grad_traj_s @ traj_grad_U_s
        cost_grad_U_t = cost_grad_traj_t @ traj_grad_U_t
        grad_cost = torch.cat((cost_grad_U_s, cost_grad_U_t), dim=-1)

        return grad_cost, cost

    def reshape_traj_grads_wrt_U(self, traj_s_grads_U: torch.Tensor, traj_t_grads_U: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for reshaping trajectory gradients to use for gradient propagation
        - Inputs:
            - traj_s_grads_U: tensor wrt U from model
            - traj_s_grads_U: tensor wrt U from model
        - Returns:
            - traj_s_grads_U: tensor wrt U for propagation
            - traj_s_grads_U: tensor wrt U for propagation
        """
        raise NotImplementedError

    def reshape_h_grads_wrt_traj(self, h_grads_traj_s: torch.Tensor, h_grads_traj_t: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for reshaping h(traj) gradients to use for gradient propagation
        - Inputs:
            - h_grads_traj_s: tensor wrt traj_s
            - h_grads_traj_t: tensor wrt traj_t
        - Returns:
            - h_grads_traj_s: tensor wrt traj_s
            - h_grads_traj_t: tensor wrt traj_t
        """
        raise NotImplementedError

    def reshape_U_for_models(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for reshaping U for model_s and model_t
        - Inputs:
            - U: tensor
        - Inputs:
            - U_s: tensor to model_s
            - U_t: tensor to model_t
        """
        raise NotImplementedError

    def reshape_trajs_for_fn(self, traj_s: torch.Tensor, traj_t: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape trajectory from model for use by h(x)
        - Inputs:
            - traj_s: tensor from model_s
            - traj_t: tensor from model_t
        - Outpus:
            - traj_s: tensor to h(traj_s, traj_t)
            - traj_t: tensor to h(traj_s, traj_t)
        """
        raise NotImplementedError