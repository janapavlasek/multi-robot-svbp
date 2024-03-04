import torch
from typing import Tuple
from abc import ABC

class DynamicsModel(ABC):
    """
    Abstract base class for all dynamics model object
    """
    def rollout(self, *args):
        raise NotImplementedError

    def rollout_w_grad(self, *args):
        raise NotImplementedError

    def is_linear(self):
        raise NotImplementedError


class PrecomputedModel(object):
    """
    A dynamics model which had previously been precomputed
    """
    def __init__(self, model: DynamicsModel) -> None:
        self.model = model
        self.stored_rollout = None
        self.stored_grads = None

    def rollout(self, *args):
        return self.stored_rollout.clone()

    def rollout_w_grad(self, *args):
        return (*[grad.clone() for grad in self.stored_grads], self.stored_rollout)

    def is_linear(self):
        return self.model.is_linear()

    def precompute_rollout(self, *args):
        self.stored_rollout = self.model.rollout(*args)

    def precompute_rollout_w_grad(self, *args):
        rollout_w_grad = self.model.rollout_w_grad(*args)
        self.stored_rollout = rollout_w_grad[-1]
        self.stored_grads = rollout_w_grad[:-1]


class LinearDynamicsModel(DynamicsModel):
    """
    Used to calculate an intermediate trajectory where control u forms part of
    the input state.
    - Given:
        - variable controls U = (...,T,u_dim)
        - variable initial state x_0 = (...,T,x_dim)
        - fixed linear model parameterized by A = (x_dim, x_dim), B = (x_dim, x_dim)
    - Returns:
        - rollout states X = (...,T+1,x_dim), where t = 0,...T
    - Implements:
        - x_k+1 = A @ x_k + B @ u_k
    """

    def __init__(self, A : torch.Tensor, B : torch.Tensor, T : int,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        Inputs:
        - A: tensor (x_dim, x_dim)
        - B: tensor (x_dim, u_dim)
        - T: int, horizon of rollout
        """
        self.tensor_kwargs = tensor_kwargs
        _, _, self._b0_A, self._b0_B = LinearDynamicsModel.create_batch_matrices(A, B, T, tensor_kwargs)

    @staticmethod
    def create_batch_matrices(A, B, T, tensor_kwargs):
        """
        Method for creating batch matrices to generate full roll out of [x0, ... xN] in one operation
        - Inputs:
            - A: tensor (x_dim, x_dim)
            - B: tensor (x_dim, u_dim)
            - T: int, horizon of rollout
        - Returns:
            - batched_A : tensor (T*x_dim, x_dim) for generating [x1, ... xN] from x0
            - batched_B : tensor (T*x_dim, T*u_dim) for generating [x1, ... xN] from x0
            - batched_0A : tensor ((T+1)*x_dim, x_dim) for generating [x0, ... xN] from x0
            - batched_0B : tensor ((T+1)*x_dim, T*u_dim) for generating [x0, ... xN] from x0
        """
        assert A.shape[-1] == A.shape[-2], f"Error A matrix, shape needs to be (x_dim,x_dim)"
        x_dim = A.shape[-1]
        u_dim = B.shape[-1]

        # batch to create [x1, ...xN]
        bA = torch.cat([torch.linalg.matrix_power(A, i + 1) for i in range(T)], dim=0)  # (x_dim*T,x_dim)
        bB = torch.eye(x_dim * T, **tensor_kwargs)
        for i in range(0, T - 1):
            bB[i * x_dim:(i + 1) * x_dim,
               (i + 1) * x_dim:] = bA.T[:, :(T - i - 1) * x_dim]
        bB = bB.T.mm(torch.block_diag(*[B for i in range(T)]))  # (x_dim*T,u_dim*T)

        # batch to create [x0, ...xN]
        b0A = torch.cat((torch.eye(x_dim, **tensor_kwargs), bA))  # (x_dim*(T+1),x_dim)
        b0B = torch.cat((torch.zeros(x_dim, u_dim * T, **tensor_kwargs), bB))  # (x_dim*(T+1),u_dim*T)

        return bA, bB, b0A, b0B

    def rollout(self, u : torch.Tensor, x_0 : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - u : tensor (...,T,u_dim) where T is the horizon
            - x_0 : tensor (...,x_dim)
        - Returns:
            - X : tensor (...,T+1,x_dim)
        """
        batch_shape = u.shape[:-2]
        T = u.shape[-2]
        u_dim = u.shape[-1]
        x_dim = x_0.shape[-1]
        return (self._b0_A @ x_0[...,None] +
                self._b0_B @ u.view(*batch_shape, T * u_dim, 1)).view(*batch_shape, T + 1, x_dim)

    def rollout_w_grad(self, u : torch.Tensor, x_0 : torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - Input:
            - u : tensor (...,T,u_dim) where T is the horizon
            - x_0 : tensor (...,x_dim)
        - Returns:
            - grad_u : tensor (...,T+1,x_dim,T,u_dim) wrt u
            - grad_x0 : tensor (...,T+1,x_dim,x_dim) wrt x0
            - X : tensor (...,T+1,x_dim)
        - NOTE:
            - grad output is different from torch.autograd.grad
            - autograd.grad accumulates the impact from all outputs into each inputs,
                resulting in summing of gradients before we can work with them
            - our grad refers to the gradient that maps between the outputs and the inputs
            - to understand treat all output variables as a single col vector
                X_out = [...,(T+1)*x_dim,1], and all inputs as a single col vector U_in = [...,T*u_dim,1] and
                X_0 = [...,x_dim,1]
            - the resulting jac for d(X_out)/d(U_in) = [...,(T+1)*x_dim,T*u_dim], and
                d(X_out)/d(X_0) = [...,(T+1)*x_dim,x_dim]
            - resulting jacs are then decomposed into the independent shapes to match original input
                and output shapes
        """
        batch_shape = u.shape[:-2]
        T, u_dim = u.shape[-2:]
        x_dim = x_0.shape[-1]
        traj = (self._b0_A @ x_0[..., None] + self._b0_B @ u.view(*batch_shape, T * u_dim, 1)
                ).view(*batch_shape, T + 1, x_dim)
        grad_u = self._b0_B.view(T + 1, x_dim, T, u_dim)
        grad_x0 = self._b0_A.view(T + 1, x_dim, x_dim)
        return grad_u.expand(*batch_shape, *grad_u.shape), grad_x0.expand(*batch_shape, *grad_x0.shape), traj

    def is_linear(self) -> bool:
        """
        Returns if the model is linear with respect to the controls and initial states
        """
        return True


class StackedTrajectoryForm(DynamicsModel):
    """
    Convinience tool that stacks the output from a dynamics model, X = f(U, x_0),
    where X = [x_1, ... x_N], u = [u_0, ... u_N-1],
    into a new form X_bar = [[x_1, u_0], ... [x_N, u_N-1]],
    effectively remapping the function as: X_bar = f'(U, x_0)
    - Given:
        - variable controls U = (...,T,u_dim)
        - variable initial state x_0 = (...,T,x_dim)
        - fixed dynamics model which outputs [x_1, ... x_N] from x_0 and [u_0, ... u_N-1]
    - Returns:
        - rollout states X_bar = [[x_1, u_0], ... [x_N, u_N-1]]
    """
    def __init__(self, dynamics_model : DynamicsModel,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Inputs:
            - dynamics_model : dynamic model with cost and cost_w_grad implementation
        """
        self.tensor_kwargs = tensor_kwargs
        self.dyn_model = dynamics_model

    def rollout(self, u : torch.Tensor, x_0 : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - u : tensor (...,T,u_dim) where T is the horizon
            - x_0 : tensor (...,x_dim)
        - Returns:
            - cost : tensor (...,T,x_dim+u_dim), where cost is X_bar = [[x_1, u_0], ... [x_N, u_N-1]]
        """
        x_traj = self.dyn_model.rollout(u, x_0)[...,1:,:] # remove x_0

        return torch.cat((x_traj, u), dim=-1)

    def rollout_w_grad(self, u : torch.Tensor, x_0 : torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - Input:
            - u : tensor (...,T,u_dim) where T is the horizon
            - x_0 : tensor (...,x_dim)
        - Returns:
            - grad_u_bar : tensor (...,T,(x_dim+u_dim),T,u_dim) wrt u
            - grad_x0_bar : tensor (...,T,(x_dim+u_dim),x_dim) wrt x0
            - cost : tensor (...,T,x_dim+u_dim), where cost is X_bar = [[x_1, u_0], ... [x_N, u_N-1]]
        """
        batch_shape = u.shape[:-2]
        T, u_dim = u.shape[-2:]
        x_dim = x_0.shape[-1]
        grad_u, grad_x0, x_traj = self.dyn_model.rollout_w_grad(u, x_0)

        x_bar_traj = torch.cat((x_traj[...,1:,:], u), dim=-1)
        extra_u_grads = torch.eye(T*u_dim, **self.tensor_kwargs) # t_1~t_N mapped to u_0~u_N-1
        extra_u_grads = extra_u_grads.view(T, u_dim, T, u_dim) # reshape
        grad_u_bar = torch.cat(
            (grad_u[...,1:,:,:,:], # remove x_0 as output
             extra_u_grads.expand(*batch_shape, T, u_dim, T, u_dim)
             ), dim=-3) # reshape and append to current grads
        grad_x0_bar = torch.cat(
            (grad_x0[...,1:,:,:], # remove x_0 as output
             torch.zeros(*batch_shape, T, u_dim, x_dim, **self.tensor_kwargs)
             ), dim=-2)

        return grad_u_bar, grad_x0_bar, x_bar_traj

    def is_linear(self) -> bool:
        """
        Returns if the model is linear with respect to the controls and initial states
        """
        return self.dyn_model.is_linear()


class LinearPointRobotModel(DynamicsModel):
    """
    """

    def __init__(self, dim, ctrl_space='acc', dt=0.1, horizon=1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}):
        """
        - Inputs:
            - dim: int, dimension of robot
            - ctrl_space: str, One of ['acc', 'vel'].
            - dt: float, time in between each time step of the rollout
            - horizon: int, number of steps to project for the rollout
            - tensor_kwargs: dict, keyword args used for generating new tensors
        """
        assert ctrl_space in ['acc', 'vel'], "Control space must be one of ['acc', 'vel']"
        self.ctrl_space = ctrl_space

        self.dim = dim
        self.x_dim = dim if ctrl_space == 'vel' else 2 * dim
        self.dt = dt
        self.T = horizon
        self.tensor_kwargs = tensor_kwargs

        # Compute the linear model matrices.
        self.A = torch.eye(self.x_dim, **tensor_kwargs)
        self.B = self.dt * torch.eye(self.dim, **tensor_kwargs)
        if ctrl_space == 'acc':
            self.A[:dim, dim:] = self.dt * torch.eye(self.dim, **tensor_kwargs)
            self.B = torch.cat((0.5 * self.dt**2 * torch.eye(self.dim, **tensor_kwargs), self.B), dim=0)

        self._linear_dyn_model = LinearDynamicsModel(self.A, self.B, horizon, tensor_kwargs)
        self._stacked_traj_form = StackedTrajectoryForm(self._linear_dyn_model, tensor_kwargs)

    def set_horizon(self, horizon):
        """
        Sets the horizon of the robot model
        - Input:
            - horizon: int, number of projection timesteps in the rollout
        """
        self.T = horizon
        self._linear_dyn_model = LinearDynamicsModel(self.A, self.B, horizon, self.tensor_kwargs)
        self._stacked_traj_form = StackedTrajectoryForm(self._linear_dyn_model, self.tensor_kwargs)

    def rollout(self, U, x_0):
        """
        - Input:
            - u : tensor (...,T,u_dim) where T is the horizon
            - x_0 : tensor (...,x_dim)
        - Returns:
            - cost : tensor (...,T,x_dim+u_dim), where cost is X_bar = [[x_1, u_0], ... [x_N, u_N-1]]
        """
        return self._stacked_traj_form.rollout(U, x_0)

    def rollout_w_grad(self, U, x_0):
        """
        - Input:
            - u : tensor (...,T,u_dim) where T is the horizon
            - x_0 : tensor (...,x_dim)
        - Returns:
            - grad_u_bar : tensor (...,T,(x_dim+u_dim),T,u_dim) wrt u
            - grad_x0_bar : tensor (...,T,(x_dim+u_dim),x_dim) wrt x0
            - cost : tensor (...,T,x_dim+u_dim), where cost is X_bar = [[x_1, u_0], ... [x_N, u_N-1]]
        """
        return self._stacked_traj_form.rollout_w_grad(U, x_0)

    def is_linear(self) -> bool:
        """
        Returns if the model is linear with respect to the controls and initial states
        """
        return True


class GaussianLinearPointRobotModel(LinearPointRobotModel):
    """
    A version of LinearPointRobotModel that takes in Gaussians instead of Non random values
    """

    def __init__(self, dim, moment_form=True, **kwargs):
        super().__init__(dim, **kwargs)
        self._rollout_fn = self.rollout_moment if moment_form else self.rollout_canonical

    def rollout(self, U, x_0):
        """
        Roll out with gaussian U and x_0.
        """
        return self._rollout_fn(*U, *x_0)

    def rollout_w_grad(self, U, x_0):
        return self._stacked_traj_form.rollout_w_grad(U, x_0)

    def rollout_moment(self,
                       U_mu: torch.Tensor, U_sigma: torch.Tensor,
                       x_0_mu: torch.Tensor, x_0_sigma: torch.Tensor):
        """
        Roll out given U, x_0 in moments form.
        Assumption: U and x_0 are unrelated -> Cov(U,x_0) = zeros
        - Inputs:
            - U_mu: tensor (...,T,u_dim), mean of U
            - U_sigma: tensor (...,T,u_dim,T,u_dim), covar of U
            - x_0_mu: tensor (...,x_dim), mean of x_0
            - x_0_sigma: tensor (...,x_dim,x_dim), covar of x_0
        - Returns:
            - traj_mu: tensor (...,T,x_dim+u_dim), mean of trajectory
            - traj_sigma: (...,T,x_dim+u_dim,T,x_dim+u_dim) covar of trajectory
        """
        batch_shape = U_mu.shape[:-2]
        grad_U, grad_x_0, traj_mu = self._stacked_traj_form.rollout_w_grad(U_mu, x_0_mu)
        # since linear, only the gradients on the variable is used in covariance calculation
        # Y'= AX'+b-> E(Y')= AE(X')+b, Cov(Y')=ACov(X')A.t, where X', Y' are gaussians
        # Y'= X1'+X2'-> E(Y')= E(X1')+E(X2'), Cov(Y')=Cov(X1')+Cov(X2')+2Cov(X1',X2'), where X1',X2',Y are gaussians
        # we treat trajectory calculation as a linear calculation where traj' = F(U',x_0')
        # since its linear and U' and x_0' are unrelated -> traj' = grad_U@U' + grad_x0@x_0' + k
        grad_U = grad_U.view(*batch_shape, self.T*(self.x_dim+self.dim), self.T*self.dim)
        U_sigma = U_mu.view(*batch_shape, self.T*self.dim, self.T*self.dim)
        grad_x_0 = grad_x_0.view(*batch_shape, self.T*(self.x_dim+self.dim), self.x_dim)
        traj_sigma = grad_U @ U_sigma @ grad_U.transpose(-1,-2) + grad_x_0 @ x_0_sigma @ grad_x_0.transpose(-1,-2)
        traj_sigma = traj_sigma.view(*batch_shape, self.T, self.x_dim+self.dim, self.T, self.x_dim+self.dim)

        return traj_mu, traj_sigma

    def rollout_canonical(self,
                          U_eta: torch.Tensor, U_lambda: torch.Tensor,
                          x_0_eta: torch.Tensor, x_0_lambda: torch.Tensor):
        """
        Roll out given U, x_0 in canonical form.
        Assumption: U and x_0 are unrelated -> Cov(U,x_0) = zeros
        - Inputs:
            - U_eta: tensor (...,T,u_dim), potential of U
            - U_lambda: tensor (...,T,u_dim,T,u_dim), precision/information matrix of U
            - x_0_eta: tensor (...,x_dim), potential of U of x_0
            - x_0_lambda: tensor (...,x_dim,x_dim), precision/information of x_0
        - Returns:
            - traj_eta: tensor (...,T,x_dim+u_dim), potential of trajectory
            - traj_lambda: (...,T,x_dim+u_dim,T,x_dim+u_dim) precision/information of trajectory
        """
        batch_shape = U_eta.shape[:-2]
        U_eta = U_eta.view(*batch_shape, self.T*self.x_dim)
        U_lambda = U_lambda.view(*batch_shape, self.T*self.x_dim, self.T*self.x_dim)
        U_mu = torch.linalg.solve(U_lambda, U_eta)
        U_sigma = torch.linalg.solve(U_lambda, torch.eye(self.dim, **self.tensor_kwargs))
        U_mu = U_mu.view(*batch_shape, self.T, self.dim)
        U_sigma = U_sigma.view(*batch_shape, self.T, self.x_dim, self.T, self.x_dim)

        x_0_mu = x_0_mu.view(*batch_shape, -1)
        x_0_lambda = x_0_lambda.view(*batch_shape, -1)
        x_0_mu = torch.linalg.solve(x_0_lambda, x_0_eta)
        x_0_sigma = torch.linalg.solve(x_0_lambda, torch.eye(self.x_dim, **self.tensor_kwargs))
        x_0_mu = x_0_mu.view(*batch_shape, self.x_dim)
        x_0_sigma = x_0_sigma.view(*batch_shape, self.T, self.x_dim, self.T, self.x_dim)

        traj_mu, traj_sigma = self.rollout_moment(U_mu, U_sigma, x_0_mu, x_0_sigma)

        traj_mu = traj_mu.view(*batch_shape, self.T*(self.x_dim+self.dim))
        traj_sigma = traj_sigma.view(*batch_shape, self.T*(self.x_dim+self.dim), self.T*(self.x_dim+self.dim))
        traj_eta = torch.linalg.solve(traj_sigma, traj_mu)
        traj_lambda = torch.linalg.solve(traj_sigma, torch.eye(self.T*(self.x_dim+self.dim), **self.tensor_kwargs))
        traj_eta = traj_eta.view(*batch_shape, self.T, self.x_dim+self.dim)
        traj_lambda = traj_lambda.view(*batch_shape, self.T, self.x_dim+self.dim, self.T, self.x_dim+self.dim)

        return traj_eta, traj_lambda
