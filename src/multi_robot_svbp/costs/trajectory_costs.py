"""
Trajectory costs
- Sets of functions commonly used for costing trajectories
- Trajectories will have the form of (...,T,num_diff * pos_dim)
- Also contains Dynamics model and commonly used trajectory forms to facilitate user with gradient propagation
"""

import torch
from typing import Iterable, Tuple, Union

from multi_robot_svbp.costs.base_costs import BaseCost, LinearCost, QuadraticCost, BoundCost


class RunningDeltaCost(BaseCost):
    """
    A common way of costing deviation from an intended state
    - Given:
        - variable state X (...,T,state_dim) tensor = [[x, x', x'', ...], ... ] over the horizon to be summed over
        - fixed intended states X_bar (T,state_dim) tensor = [x_bar, x_bar', x_bar'', ...]
            where x' represents dx/dt
    - Returns:
        - scalar cost
    """
    def __init__(self, Qs : Iterable[torch.Tensor], x_bars : Union[None,Iterable[torch.Tensor]], sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Input:
            - Qs : [Q_0, Q_1, ...], list of Qi costs, up to the derivative covered
            - x_bars : None or [x_bar, x_bar', ...], list of intended state x_bar, up to the derivative covered
                if None costs against change in pose and 0 vel 0 accel
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        Q = torch.block_diag(*Qs)
        self.quad_cost = QuadraticCost(Q, sigma=sigma, tensor_kwargs=tensor_kwargs)
        if x_bars is None:
            self._dim_0 = Qs[0].shape[-1]
        else:
            self._X_bar = torch.cat(x_bars).to(**tensor_kwargs)
        self._get_x_bar = self._get_x_bar_not_defined if x_bars is None else self._get_x_bar_defined

    def cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x : tensor (...,T,x_dim) where T is the horizon to be summed over
        - Returns:
            - cost : tensor (...,)
        """
        X_bar = self._get_x_bar(x)
        return self.quad_cost.cost(x - X_bar).sum(dim=-1)

    def grad_w_cost(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Input:
            - x : tensor (...,T,x_dim) where T is the horizon to be summed over
        - Returns:
            - grad : tensor (...,T,x_dim)
            - cost : tensor (...,)
        """
        X_bar = self._get_x_bar(x)
        grad, cost = self.quad_cost.grad_w_cost(x - X_bar)  # y = x-x_bar => dy/dx = 1
                                                            # y = f1(x1)+f2(x2)+ ...
                                                            # => dy/dX = [df1/dx1, df2/dx2, ...]
        return grad, cost.sum(dim=-1)

    def _get_x_bar_defined(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method for getting X_bar if it had been defined by user
        """
        return self._X_bar

    def _get_x_bar_not_defined(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method for getting X_bar if it had not been defined by user.
        In this case, cost against changing starting pose (x_current, 0 vel, 0)
        """
        x_bar = x.clone()
        x_bar[self._dim_0:] = 0
        return x_bar

class TerminalDeltaCost(BaseCost):
    """
    A common way of costing the final state with respect to an intended state
    - Given:
        - variable state X (...,T,state_dim) tensor = [[x, x', x'', ...], ... ]
            of which only the last point in T is cost
        - fixed intended states X_bar (T,state_dim) tensor = [x_bar, x_bar', x_bar'', ...]
            where x' represents dx/dt
    - Returns:
        - scalar cost
    """
    def __init__(self, Qs : Iterable[torch.Tensor], x_bars : Iterable[torch.Tensor], sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Input:
            - Qs : [Q_0, Q_1, ...], list of Qi costs, up to the derivative covered
            - x_bars : [x_bar, x_bar', ...], list of intended state x_bar, up to the derivative covered
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        Q = torch.block_diag(*Qs)
        self.quad_cost = QuadraticCost(Q, sigma=sigma, tensor_kwargs=tensor_kwargs)
        self.X_bar = torch.cat(x_bars).to(**tensor_kwargs)

    def cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x : tensor (...,T,x_dim) where T is the horizon
        - Returns:
            - cost : tensor (...,)
        """
        return self.quad_cost.cost(x[...,-1,:] - self.X_bar)

    def grad_w_cost(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Input:
            - x : tensor (...,T,x_dim) where T is the horizon
        - Returns:
            - grad : tensor (...,T,x_dim)
            - cost : tensor (...,)
        """
        grad_N, cost = self.quad_cost.grad_w_cost(x[...,-1,:] - self.X_bar)
        grad = torch.zeros_like(x)
        grad[...,-1,:] = grad_N # y = x-x_bar => dy/dx = 1
                                # y = fN(xN) => dy/dX = [0, 0, ..., dfN/dxN]
        return grad, cost

class RunningCollisionCost(BaseCost):
    """
    Generic collision cost function that costs X1, X2 for being too close to each other
    - Given:
        - variable states X1, X2 = [[x, x', x'', ...], ... ] over the horizon to be summed over
            and x' represents dx/dt,
        - fixed radius float, above which there's no penalty,
        - fixed k_bend float, bending factor that bends the cost function to be
            linear (k=1), convex (k=0~1) or concave (k=1~inf),
        - fixed sigma_T (T,) tensor or float, represents the scaling for each timestep from
            t=0 to t=T, if tensor is given, fixes horizon of inputs to size T
    - Returns:
        - scalar cost
    """
    def __init__(self, pos_dim : int, radius : float, k_bend : float, sigma_T : Union[torch.Tensor, float],
                 sigma=1.0, tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Input:
            - pos_dim : int, the number of dimension that encodes the position, where pos = x[...,:pos_dim]
            - radius : float, value above which no cost is applied, below which incurs the corresponding cost
            - x_bend : float, bending factor that bends the cost function to be
                linear (k=1), convex (k=(0,1]) or concave (k=[1~inf))
            - sigma_T : (T,) tensor or float, if tensor is given, scales each time component seperately and
                input will be fixed to horizons of size T
            - k : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        self.pos_dim = pos_dim
        self.radius = radius
        self.k_bend = k_bend
        self.sigma_T = sigma_T.to(**tensor_kwargs) if isinstance(sigma_T, torch.Tensor) \
                        else torch.tensor(sigma_T, **tensor_kwargs)
        self.sigma = sigma

    def cost(self, x1 : torch.Tensor, x2 : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x1 : tensor (...,T,x_dim) where T is the horizon
            - x2 : tensor (...,T,x_dim)
        - Returns:
            - cost : tensor (...,)
        """
        _, _, unscaled_cost_t = self._cal_intermediate_vals(x1, x2)
        return self.sigma * (self.sigma_T * unscaled_cost_t.clamp(min=0)).sum(dim=-1) # only cost those above radius

    def grad_w_cost(self, x1 : torch.Tensor, x2 : torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - Input:
            - x1 : tensor (...,T,x_dim) where T is the horizon
            - x2 : tensor (...,T,x_dim)
        - Returns:
            - grad_x1 : tensor (...,T,x_dim)
            - grad_x2 : tensor (...,T,x_dim)
            - cost : tensor (...,)
        """
        delta, dist_sqr, unscaled_cost_t = self._cal_intermediate_vals(x1, x2)

        # dense implementation
        sel_matrix = (unscaled_cost_t > 0).to(unscaled_cost_t)
        pos_grad_x1 = sel_matrix[..., None] * -self.k_bend * self.sigma_T[:,None] * \
            dist_sqr.pow(0.5 * self.k_bend - 1)[...,None] * delta / self.radius ** self.k_bend
        grad_x1 = torch.zeros_like(x1)
        grad_x1[...,:self.pos_dim] = pos_grad_x1
        grad_x2 = -grad_x1

        # # sparse implementation
        # sel_matrix = (unscaled_cost_t > 0).to(unscaled_cost_t).to_sparse()
        # sel_matrix = torch.stack([sel_matrix for _ in range(self.pos_dim)], dim=-1) # needs manual broadcast
        #                                                                             # that incurs copy
        # pos_grad_x1 = sel_matrix * -self.k_bend * self.sigma_T[:,None] * \
        #   dist_sqr.pow(0.5 * self.k_bend - 1)[...,None] * delta / self.radius ** self.k_bend
        # grad_x1 = torch.zeros_like(x1)
        # grad_x1[...,:self.pos_dim] = pos_grad_x1.to_dense()
        # grad_x2 = -grad_x1

        return (self.sigma * grad_x1,
                self.sigma * grad_x2,
                self.sigma * (self.sigma_T * unscaled_cost_t.clamp(min=0)).sum(dim=-1))

    def _cal_intermediate_vals(self, x1 : torch.Tensor, x2 : torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates all intermediate values which optionally might be required depending
        on whether gradient calculation is triggered
        - Input:
            - x1 : tensor (...,T,x_dim) where T is the horizon
            - x2 : tensor (...,T,x_dim)
        - Returns:
            - delta : tensor (...,T,pos_dim), delta of pose only
            - dist_sqr : tensor (...,T), dist_sqr at for each pose
            - unscaled_cost_t : tensor (...,T), unscaled costs for each time point in each trajectory
        """
        delta = x1[...,:self.pos_dim] - x2[...,:self.pos_dim]
        dist_sqr = (delta[...,None,:] @ delta[...,:,None])[...,0,0] # ensures values are positive
        unscaled_cost_t = 1 - dist_sqr.pow(0.5 * self.k_bend) / self.radius ** self.k_bend

        return delta, dist_sqr, unscaled_cost_t


class RunningCrossCollisionCost(BaseCost):
    """
    Similar collision cost function to RunningCollisionCost, but differ in terms of number of comparisons made.
    RunningCollisionCost takes in X1 = (...,T,pos_dim) tensor and X2 = (...,T,pos_dim) tensor
    and returns C = (...,) tensor.
    RunningCrossCollision takes in X1 = (...,K1,T,pos_dim) tensor and X2 = (...,K2,T,pos_dim) tensor
    and returns C = (...,K1,K2) tensor.
    I.E. RunningCrossCollision compares each trajectory from X1/X2 and compare with EVERY other trajectory
    from X2/X1.
    - Given:
        - variable states X1, X2 = [[x, x', x'', ...], ... ] over the horizon to be summed over
            and x' represents dx/dt,
        - fixed radius float, above which there's no penalty,
        - fixed k_bend float, bending factor that bends the cost function to be
            linear (k=1), convex (k=0~1) or concave (k=1~inf)
        - fixed sigma_T (T,) tensor or float, represents the scaling for each timestep from
            t=0 to t=T, if tensor is given, fixes horizon of inputs to size T
    - Returns:
        - scalar cost
    """
    def __init__(self, pos_dim : int, radius : float, k_bend : float, sigma_T : Union[torch.Tensor, float],
                 sigma=1.0, tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        Input:
        - pos_dim : int, the number of dimension that encodes the position, where pos = x[...,:pos_dim]
        - radius : float, value above which no cost is applied, below which incurs the corresponding cost
        - x_bend : float, bending factor that bends the cost function to be linear (k=1),
                    convex (k=(0,1]) or concave (k=[1~inf))
        - sigma_T : (T,) tensor or float, if tensor is given, scales each time component seperately and
                    input will be fixed to horizons of size T
        - k : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        self.pos_dim = pos_dim
        self.radius = radius
        self.k_bend = k_bend
        self.sigma_T = sigma_T.to(**tensor_kwargs) if isinstance(sigma_T, torch.Tensor) else torch.as_tensor(sigma_T, **tensor_kwargs)
        self.sigma = sigma

    def cost(self, x1 : torch.Tensor, x2 : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x1 : tensor (...,K1,T,x_dim) where T is the horizon
            - x2 : tensor (...,K2,T,x_dim)
        - Returns:
            - cost : tensor (...,K1,K2)
        """
        _, _, unscaled_cost_t = self._cal_intermediate_vals(x1, x2)
        return self.sigma * (self.sigma_T  * unscaled_cost_t.clamp(min=0)).sum(dim=-1) # only cost those above radius

    def grad_w_cost(self, x1 : torch.Tensor, x2 : torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - Input:
            - x1 : tensor (...,K1,T,x_dim) where T is the horizon
            - x2 : tensor (...,K2,T,x_dim)
        - Returns:
            - grad_x1 : tensor (...,K1,K2,T,x_dim) # NOTE: usually gradients are in (...,output,input)
                                                    but this convention is broken here
                                                    -> (input is supposed to be ...K1,T,pos_dim)
            - grad_x2 : tensor (...,K1,K2,T,x_dim)
            - cost : tensor (...K1,K2)
        """
        batch_shape = x1.shape[:-3]
        K1, T, x_dim = x1.shape[-3:]
        K2 = x2.shape[-3]

        delta, dist_sqr, unscaled_cost_t = self._cal_intermediate_vals(x1, x2)

        # dense implementation
        sel_matrix = (unscaled_cost_t > 0).to(unscaled_cost_t)
        pos_grad_x1 = sel_matrix[..., None] * -self.k_bend * self.sigma_T[:,None] \
            * dist_sqr.pow(0.5 * self.k_bend - 1)[...,None] * delta / self.radius ** self.k_bend
        grad_x1 = torch.zeros(*batch_shape, K1, K2, T, x_dim, **self.tensor_kwargs)
        grad_x1[...,:self.pos_dim] = pos_grad_x1
        grad_x2 = -grad_x1

        # # sparse implementation
        # sel_matrix = (unscaled_cost_t > 0).to(unscaled_cost_t).to_sparse()
        # sel_matrix = torch.stack(
        #     [sel_matrix for _ in range(self.pos_dim)], dim=-1) # needs manual broadcast that incurs copy
        # pos_grad_x1 = sel_matrix * -self.k_bend * self.sigma_T[:,None] \
        #     * dist_sqr.pow(0.5 * self.k_bend - 1)[...,None] * delta / self.radius ** self.k_bend
        # grad_x1 = torch.zeros(*batch_shape, K1, K2, T, x_dim, **self.tensor_kwargs)
        # grad_x1[...,:self.pos_dim] = pos_grad_x1.to_dense()
        # grad_x2 = -grad_x1

        return self.sigma * grad_x1, self.sigma * grad_x2, self.sigma * \
            (unscaled_cost_t.clamp(min=0) * self.sigma_T).sum(dim=-1)

    def _cal_intermediate_vals(self, x1 : torch.Tensor, x2 : torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates all intermediate values which optionally might be required depending on
        whether gradient calculation is triggered.
        - Input:
            - x1 : tensor (...,K1,T,x_dim) where T is the horizon
            - x2 : tensor (...,K2,T,x_dim)
        - Returns:
            - delta : tensor (...,K1,K2,T,pos_dim), delta of pose only
            - dist_sqr : tensor (...,K1,K2,T), dist_sqr at for each pose
            - unscaled_cost_t : tensor (...,K1,K2,T), unscaled costs for each time point in each trajectory
        """
        delta = x1[...,:,None,:,:self.pos_dim] - x2[...,None,:,:,:self.pos_dim] # (...,K1,K2,T,pos_dim)
        dist_sqr = (delta[...,None,:] @ delta[...,:,None])[...,0,0] # ensures values are positive (...,K1,K2,T)
        unscaled_cost_t = 1 - dist_sqr.pow(0.5 * self.k_bend) / self.radius ** self.k_bend # (...,K1,K2,T)

        return delta, dist_sqr, unscaled_cost_t


class RunningDeltaJacCost(BaseCost):
    """
    Similar to Running Delta Cost, but instead use the jacobians of the running cost.

    This is useful for Linear Gaussian BP since it does the following:
        - Breaks singularity that results from having multiple solutions that can fulfill the lower dimension cost.
        - Instead of equating h(x) = 0 which for which a solution may not exist depending on robot distance from goal
            (not physically possible), instead sets dh(x)/d(x) = 0 which gives the x that minimizes h(x) subjected to
            Q being PSD

    NOTE: if x is obtained from a model, you cannot blindly propagate the gradient by multiplying jacobians since
    this cost is already a jacobian
    - Given:
        - variable state X (...,T,state_dim) tensor = [[x, x', x'', ...], ... ] over the horizon to be summed over
        - fixed intended states X_bar (T,state_dim) tensor = [x_bar, x_bar', x_bar'', ...]
            where x' represents dx/dt
    - Returns:
        - (...,T,state_dim,T,state_dim) cost
    """
    def __init__(self, Qs : Iterable[torch.Tensor], x_bars : Union[None,Iterable[torch.Tensor]], sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Input:
            - Qs : [Q_0, Q_1, ...], list of Qi costs, up to the derivative covered
            - x_bars : None or [x_bar, x_bar', ...], list of intended state x_bar, up to the derivative covered
                if None costs against change in pose and 0 vel 0 accel
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        Q = torch.block_diag(*Qs)
        self.linear_cost = LinearCost(Q+Q.T, sigma=sigma, tensor_kwargs=tensor_kwargs)
        if x_bars is None:
            self._dim_0 = Qs[0].shape[-1]
        else:
            self._X_bar = torch.cat(x_bars).to(**tensor_kwargs)
        self._get_x_bar = self._get_x_bar_not_defined if x_bars is None else self._get_x_bar_defined

    def cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x : tensor (...,T,x_dim) where T is the horizon to be summed over
        - Returns:
            - cost : tensor (...,T,x_dim)
        """
        X_bar = self._get_x_bar(x)
        return self.linear_cost.cost(x - X_bar)

    def grad_w_cost(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Input:
            - x : tensor (...,T,x_dim) where T is the horizon to be summed over
        - Returns:
            - grad : tensor (...,T,x_dim,T,x_dim)
            - cost : tensor (...,T,x_dim)
        """
        T = x.shape[-2]
        X_bar = self._get_x_bar(x)
        grad, cost = self.linear_cost.grad_w_cost(x - X_bar)
        # grad -> (...,T,x_dim,x_dim) -> (...,T,x_dim,T,x_dim)
        grad = torch.eye(T, **self.tensor_kwargs)[:,None,:,None] @ grad[...,None,:]
        return grad, cost

    def _get_x_bar_defined(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method for getting X_bar if it had been defined by user
        """
        return self._X_bar

    def _get_x_bar_not_defined(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method for getting X_bar if it had not been defined by user.
        In this case, cost against changing starting pose (x_current, 0 vel, 0)
        """
        x_bar = x.clone()
        x_bar[self._dim_0:] = 0
        return x_bar


class StateBoundsCost(BaseCost):
    def __init__(self, dim, pos_lims=None, max_vel=None, max_acc=None, c_pos=1, c_vel=1, c_acc=1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}):

        super().__init__(tensor_kwargs)
        if pos_lims is None:
            pos_lims = [[0] * dim, [0] * dim]
            c_pos = 0
        if max_vel is None:
            max_vel = 0
            c_vel = 0
        if max_acc is None:
            max_acc = 0
            c_acc = 0

        self.pos_bounds = BoundCost(pos_lims, sigma=c_pos, tensor_kwargs=tensor_kwargs)
        self.vel_bounds = BoundCost([[0], [max_vel**2]], sigma=c_vel, tensor_kwargs=tensor_kwargs)
        self.acc_bounds = BoundCost([[0], [max_acc**2]], sigma=c_acc, tensor_kwargs=tensor_kwargs)

        self.dim = dim

    def cost(self, x: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - x : tensor (..., T, x_dim)
        - Returns:
            - cost : tensor (...,)
        """
        pos, vel, acc = x[..., :self.dim], x[..., self.dim:2 * self.dim], x[..., 2 * self.dim:]
        vel = (vel * vel).sum(-1).unsqueeze(-1)
        acc = (acc * acc).sum(-1).unsqueeze(-1)

        cost = self.pos_bounds(pos)
        cost += self.vel_bounds(vel)
        cost += self.acc_bounds(acc)
        return cost.sum(-1)  # Sum over the horizon: (..., T) -> (...,)
