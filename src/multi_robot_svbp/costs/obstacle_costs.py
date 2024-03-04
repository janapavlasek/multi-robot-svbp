"""
Obstacle cost functions
- Commonly used to calculate breaching of obstacle given a trajectory
- Only 2D cases now
"""

import torch
from typing import Iterable, Tuple, Union
from multi_robot_svbp.costs.base_costs import BaseCost, CompositeMaxCost

class AABBDistance2DCost(BaseCost):
    """
    Cost as a signed distance sqaured from the edges of a 2D Axis-Aligned Bounding Box (AABB)
    - Given:
        - variable state X (...,2) tensor, only accepts 2D inputs since calculation is meant for 2D objects only
        - fixed width float, extends of box in 1st axis (x_extends)
        - fixed height float, extends of box in 2nd axis (y_extends)
        - fixed center (2,) tensor, coordinate of centre of box (x,y)
    - Returns:
        - scalar cost as distance squared from box edge, -ve outside box, +ve inside box
    """
    def __init__(self, width : float, height : float, center : torch.Tensor, sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Input:
            - width : float, extends of box in 1st axis (x_extends)
            - height : float, extends of box in 2nd axis (y_extends)
            - center : (2,) tensor, coordinate of centre of box (x,y)
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        self.half_extends = torch.tensor([width / 2, height / 2], **tensor_kwargs)
        self.center = center.to(**tensor_kwargs)
        self.sigma = sigma
        # dont store width, height to reduce confusion if user tries to
        # change them without calculating half_extends

    def cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x : tensor (...,2)
        - Returns:
            - cost : tensor (...,)
        """
        x_rel = x - self.center
        q = (torch.abs(x_rel) - self.half_extends)
        dist_out = torch.sum(q.clamp(min=0)**2, dim=-1)
        dist_in = q.max(dim=-1)[0].clamp(max=0)**2
        return self.sigma * (dist_in - dist_out)

    def grad_w_cost(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Input:
            - x : tensor (...,2)
        - Returns:
            - grad : tensor (...,2)
            - cost : tensor (...,)
        """
        x_rel = x - self.center
        q = (torch.abs(x_rel) - self.half_extends)
        dist_out = torch.sum(q.clamp(min=0)**2, dim=-1)
        q_in, q_in_ind = q.max(dim=-1)
        dist_in = q_in.clamp(max=0)**2
        cost = self.sigma * (dist_in - dist_out)

        dq_dx = x_rel.sign()
        ddout_dq = (2 * q).clamp(min=0)
        ddin_dq = 2 * (torch.eye(2, **self.tensor_kwargs)[q_in_ind] * q).clamp(max=0)
        grad = self.sigma * dq_dx * (ddin_dq - ddout_dq)

        return grad, cost

class CircleDistance2DCost(BaseCost):
    """
    Cost as a signed distance sqaured from the edges of a 2D circle
    - Given:
        - variable state X (...,2) tensor, only accepts 2D inputs since calculation is meant for 2D objects only
        - fixed radius float, radius of the circle
        - fixed center (2,) tensor, coordinate of centre of circle (x,y)
    - Returns:
        - scalar cost as distance squared from box edge, -ve outside circle, +ve inside circle
    """
    def __init__(self, radius : float, center : torch.Tensor, sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        Input:
        - radius : float, radius of the circle
        - center : (2,) tensor, coordinate of centre of box (x,y)
        - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        self.radius = radius
        self.center = center.to(**tensor_kwargs)
        self.sigma = sigma

    def cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x : tensor (...,2)
        - Returns:
            - cost : tensor (...,)
        """
        x_rel = x - self.center
        s_dist = self.radius - x_rel.norm(dim=-1)
        return self.sigma * s_dist.sign() * s_dist.pow(2)

    def grad_w_cost(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Input:
            - x : tensor (...,2)
        - Returns:
            - grad : tensor (...,2)
            - cost : tensor (...,)
        """
        x_rel = x - self.center
        x_rel_norm2 = x_rel.square().sum(dim=-1)
        x_rel_norm = x_rel_norm2.sqrt()
        s_dist = self.radius - x_rel_norm
        sign = s_dist.sign()
        d_2 = s_dist.sign() * s_dist.square()

        grad = -2 * sign[...,None] * s_dist[...,None] / x_rel_norm[...,None] * x_rel

        return self.sigma * grad, self.sigma * d_2


class SignedDistanceMap2DCost(BaseCost):
    """
    Implements the maximum signed distance sqr function given a set of 2D costs fn
    - Given:
        - variable state X (...,T,2) tensor, where T represents the horizon
        - fixed 2D distance costs objects
    - Returns:
        - scalar cost as distance squared from edge, -ve outside all obstacles, +ve inside any obstacle
    """
    def __init__(self, dist_2d_costs: Iterable[Union[AABBDistance2DCost, CircleDistance2DCost]],
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        super().__init__(tensor_kwargs)
        self.max_sd_cost = CompositeMaxCost(dist_2d_costs, tensor_kwargs=tensor_kwargs)

    def cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x : tensor (...,T,x_dim)
        - Returns:
            - cost : tensor (...,T)
        """
        return self.max_sd_cost.cost(x)

    def grad_w_cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x : tensor (...,T,x_dim)
        - Returns:
            - grad : tensor (...,T,x_dim)
            - cost : tensor (...,T)
        """
        return self.max_sd_cost.grad_w_cost(x)

    def add_aabb(self, width : float, height : float, center : torch.Tensor, sigma=1.0) -> None:
        """
        Inserts a Axis-Aligned Bounding Box (AABB) into the map
        - Input:
            - width : float, extends of box in 1st axis (x_extends)
            - height : float, extends of box in 2nd axis (y_extends)
            - center : (2,) tensor, coordinate of centre of box (x,y)
            - sigma : float, scalar term to scale the impact of this cost
        - NOTE:
            - becareful when adjusting values of sigma != 1.0 as it will not correspond to the signed distance anymore
        """
        self.max_sd_cost.costs.append(AABBDistance2DCost(width,height,center,sigma=sigma,tensor_kwargs=self.tensor_kwargs))

    def add_circle(self, radius : float, center : torch.Tensor, sigma=1.0) -> None:
        """
        Inserts a circle into the map
        - Input:
            - width : float, extends of box in 1st axis (x_extends)
            - height : float, extends of box in 2nd axis (y_extends)
            - center : (2,) tensor, coordinate of centre of box (x,y)
            - k : float, scalar term to scale the impact of this cost
        - NOTE:
            - becareful when adjusting values of sigma != 1.0 as it will not correspond to the signed distance anymore
        """
        self.max_sd_cost.costs.append(CircleDistance2DCost(radius,center,sigma=sigma,tensor_kwargs=self.tensor_kwargs))


class ExponentialSumObstacleCost(BaseCost):
    """
    Cost as a exponent of the violation or clearance,
    has 2 scaled components, the negative portion (not in collision) and the positive portion (in collision),
    summed accross the entire trajectory
    - Given:
        - variable state X (...,T,x_dim) tensor, where T represents the horizon,
            x_dim represent the state dim where pos
        - fixed SignedDistMap2DCost
    - Returns:
        - scalar cost as distance squared from box edge, -ve outside circle, +ve inside circle
    - Implements:
        - cost = sigma * sum_T{ exp(sigma_obs_in * max(sd_cost,0) + sigma_obs_out * min(sd_cost,0))}
    """
    def __init__(self, signed_2d_map : SignedDistanceMap2DCost,
                 sigma_obs_out : float, sigma_obs_in : float, sigma=1,
                 dim_0 = 0, dim_1 = 1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Inputs:
            - signed_2d_map : SignedDistanceMap2DCost used for calculating signed distances
            - sigma_obs_out : float, scaling factor when closer to an obstacle
            - sigma_obs_in : float, scaling factor when breaching an obstacle
            - sigma : float, scaling factor for entire cost
            - dim_0 : int, dim that encodes the first position dimension in the incoming trajectory
            - dim_1 : int, dim that encodes the second position dimension in the incoming trajectory
        """
        super().__init__(tensor_kwargs)
        self.max_sd_cost = signed_2d_map
        self.sigma_obs_out = sigma_obs_out
        self.sigma_obs_in = sigma_obs_in
        self.sigma = sigma
        self.dim_0 = dim_0
        self.dim_1 = dim_1

    def cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x : tensor (...,T,x_dim)
        - Returns:
            - cost : tensor (...,)
        """
        max_sd_cost = self.max_sd_cost.cost(x[...,[self.dim_0, self.dim_1]])
        log_cost_i = self.sigma_obs_out * max_sd_cost.clamp(max=0) + self.sigma_obs_in * max_sd_cost.clamp(min=0)
        return self.sigma * log_cost_i.exp().sum(dim=-1)

    def grad_w_cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x : tensor (...,T,x_dim)
        - Returns:
            - grad : tensor (...,T,x_dim)
            - cost : tensor (...,)
        """
        max_sd_grad, max_sd_cost = self.max_sd_cost.grad_w_cost(x[...,[self.dim_0, self.dim_1]])
        max_sd_pos = max_sd_cost > 0
        log_cost_i = self.sigma_obs_out * ~max_sd_pos * max_sd_cost + \
            self.sigma_obs_in * max_sd_pos * max_sd_cost
        cost_i = log_cost_i.exp()
        log_grad_01 = self.sigma_obs_out * ~max_sd_pos[...,None] * max_sd_grad + \
            self.sigma_obs_in * max_sd_pos[...,None] * max_sd_grad
        grad_01 = log_grad_01 * cost_i[...,None]
        grad = torch.zeros_like(x)
        grad[...,[self.dim_0, self.dim_1]] = grad_01
        return self.sigma * grad, self.sigma * cost_i.sum(dim=-1)


class KBendingObstacleCost(BaseCost):
    """
    Obstacle cost that scales between [0-inf) for violation with an obstacle with a bending factor that
    allows user to adjust this curve.
    - Given:
        - variable states X1, X2 = [[x, x', x'', ...], ... ] over the horizon to be summed over
            and x' represents dx/dt,
        - fixed radius float, above which there's no penalty,
        - fixed k_bend float, bending factor that bends the cost function to be
            linear (k=1), convex (k=1~inf) or concave (k=0~1),
        - fixed sigma_T (T,) tensor or float, represents the scaling for each timestep from
            t=0 to t=T, if tensor is given, fixes horizon of inputs to size T
    - Returns:
        - scalar cost
    - Implements: c = sigma * sigma_T * min{0, s_dist+r}^k
    """
    def __init__(self,
                 signed_2d_map : SignedDistanceMap2DCost,
                 radius : float, k_bend : float, sigma_T : Union[torch.Tensor, float],
                 sigma=1.0,
                 dim_0 = 0, dim_1 = 1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Input:
            - signed_2d_map : SignedDistanceMap2DCost used for calculating signed distances
            - radius : float, radius we expect the point to be away from obstacles
            - k_bend : float, bending factor that bends the cost function to be
                linear (k=1), convex (k=(0,1]) or concave (k=[1~inf))
            - sigma_T : (T,) tensor or float, if tensor is given, scales each time component seperately and
                input will be fixed to horizons of size T
            - sigma : float, scalar term to scale the impact of this cost
            - dim_0 : int, dim that encodes the first position dimension in the incoming trajectory
            - dim_1 : int, dim that encodes the second position dimension in the incoming trajectory
        """
        super().__init__(tensor_kwargs)
        assert radius >= 0, "input radius is not positive."
        self.max_sd_cost = signed_2d_map
        self.radius = radius
        self.k_bend = k_bend
        self.sigma_T = sigma_T.to(**tensor_kwargs) if isinstance(sigma_T, torch.Tensor) \
                        else torch.tensor(sigma_T, **tensor_kwargs)
        self.sigma = sigma
        self.dim_0 = dim_0
        self.dim_1 = dim_1

    def cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Input:
            - x : tensor (...,T,x_dim) where T is the horizon
        - Returns:
            - cost : tensor (...,T)
        """
        signed_dist_sqr = self.max_sd_cost.cost(x[...,[self.dim_0, self.dim_1]])
        signed_dist = signed_dist_sqr.sign() * signed_dist_sqr.abs().sqrt() + self.radius
        return self.sigma * self.sigma_T * signed_dist.clamp(min=0).pow(self.k_bend)

    def grad_w_cost(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Input:
            - x : tensor (...,T,x_dim) where T is the horizon
        - Returns:
            - grad : tensor (...,T,T,x_dim)
            - cost : tensor (...,T)
        """
        T = x.shape[-2]
        grad_signed_dist_sqr, signed_dist_sqr = self.max_sd_cost.grad_w_cost(x[...,[self.dim_0, self.dim_1]])

        signed_dist = signed_dist_sqr.sign() * signed_dist_sqr.abs().sqrt() + self.radius
        cost = self.sigma * self.sigma_T * signed_dist.clamp(min=0).pow(self.k_bend)

        non_positive = signed_dist <= 0
        grad_signed_dist_sqr[non_positive] *= 0
        grad_signed_dist_sqr[~non_positive] = self.k_bend * signed_dist[~non_positive].pow(self.k_bend - 1)[...,None] * \
            0.5 * signed_dist_sqr[~non_positive].abs().pow(-0.5)[...,None] * grad_signed_dist_sqr[~non_positive]
        grad_cost = torch.zeros_like(x)
        grad_cost[...,[self.dim_0, self.dim_1]] = self.sigma * self.sigma_T[...,None] * grad_signed_dist_sqr
        grad_cost = torch.eye(T, **self.tensor_kwargs)[...,None] @ grad_cost[...,None,:]

        return grad_cost, cost
