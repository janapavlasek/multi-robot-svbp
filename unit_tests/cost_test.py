"""
Test codes for testing costs related modules
"""

from typing import Tuple
import torch

from multi_robot_svbp.costs.base_costs import *
from multi_robot_svbp.costs.trajectory_costs import *
from multi_robot_svbp.costs.obstacle_costs import *

from torch_bp.util.misc import euclidean_distance

import warnings


def test_base_cost():

    class TrivialCost(BaseCost):
        def __init__(self, tensor_kwargs=...) -> None:
            super().__init__(tensor_kwargs)

        def cost(self, x : torch.Tensor) -> torch.Tensor:
            return x.sum()

        def grad_w_cost(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.ones_like(x), x.sum()

    batch_size = 3
    x_dim = 4

    trivial_cost = TrivialCost()
    x = torch.randn(batch_size, x_dim, requires_grad=True)
    cost = trivial_cost(x)
    grad, cost2 = trivial_cost(x, grad=True)
    grad_auto, = torch.autograd.grad(cost.sum(), x)

    assert True # successfully used __call__


def test_linear_cost():
    batch_size = 3
    q_dim = 4
    sigma = 1.2

    Q = torch.randn(q_dim, q_dim)
    linear_cost = LinearCost(Q, sigma)
    x = torch.randn(batch_size, q_dim, requires_grad=True)
    grad, cost = linear_cost.grad_w_cost(x)
    grad_auto = torch.vmap(torch.func.jacrev(linear_cost.cost))(x)

    assert torch.allclose(grad, grad_auto), "LinearCost implemented grad not same as auto grad!"

def test_quadratic_cost():
    batch_size = 4
    q_dim = 2
    sigma = 2.3

    Q = torch.randn(q_dim, q_dim)
    quad_cost = QuadraticCost(Q, sigma)
    x = torch.randn(batch_size, q_dim, requires_grad=True)
    grad, cost = quad_cost.grad_w_cost(x)
    grad_auto, = torch.autograd.grad(cost.sum(), x)

    assert torch.allclose(grad, grad_auto), "QuadraticCost implemented grad not same as auto grad!"


def test_composite_max_cost():
    batch_size = 5
    base_dim = 3
    num_diff = 4
    T = 2

    Qs_rc = [torch.randn(base_dim, base_dim) for _ in range(num_diff)]
    x_bars_rc = [torch.randn(base_dim) for _ in range(num_diff)]
    sigma_rc = 3.2
    Qs_tc = [torch.randn(base_dim, base_dim) for _ in range(num_diff)]
    x_bars_tc = [torch.randn(base_dim) for _ in range(num_diff)]
    opt_scalar = 52.
    sigma_tc = 30.4

    sigma_max = 1.2


    running_delta_cost = RunningDeltaCost(Qs_rc, x_bars_rc, sigma=sigma_rc)
    terminal_delta_cost = TerminalDeltaCost(Qs_tc, x_bars_tc, sigma=sigma_tc)

    max_cost = CompositeMaxCost((running_delta_cost, terminal_delta_cost),opt_scalar, sigma=sigma_max)
    x = torch.randn(batch_size, T, base_dim * num_diff, requires_grad=True)
    grad, cost = max_cost.grad_w_cost(x)
    grad_auto, = torch.autograd.grad(cost.sum(), x)

    assert torch.allclose(grad, grad_auto, atol=3e-6), "CompositeMaxCost implemented grad not same as auto grad!"
    if not torch.allclose(grad, grad_auto):
        warnings.warn(f"CompositeMaxCost implemented grad deviates from auto grad value beyond torch.allclose default tolerances!!")


def test_composite_sum_cost():
    batch_size = 4
    base_dim = 3
    num_diff = 4
    pos_dim = 2
    T = 3

    radius_1 = 0.3
    k_bend_1 = 0.1
    sigma_T_1 = torch.linspace(20, 2, T)
    sigma_1 = 4.3
    radius_2 = 0.24
    k_bend_2 = 0.5
    sigma_2 = 3.8
    add_scalar = 4.2
    sigma_T_2 = torch.linspace(100, 10, T)
    sigma_sum = 2.4

    runing_collision_cost_1 = RunningCollisionCost(pos_dim, radius_1, k_bend_1, sigma_T_1, sigma=sigma_1)
    runing_collision_cost_2 = RunningCollisionCost(pos_dim, radius_2, k_bend_2, sigma_T_2, sigma=sigma_2)

    sum_cost = CompositeSumCost((runing_collision_cost_1, runing_collision_cost_2), add_scalar, sigma=sigma_sum)
    x1 = torch.randn(batch_size, T, base_dim * num_diff, requires_grad=True)
    x2 = torch.randn_like(x1, requires_grad=True)
    grad_1, grad_2, cost = sum_cost.grad_w_cost(x1, x2)
    grad_1_auto, grad_2_auto = torch.autograd.grad(cost.sum(), (x1, x2))

    assert torch.allclose(grad_1, grad_1_auto, atol=3e-6), "CompositeSumCost implemented grad_1 not same as auto grad!"
    assert torch.allclose(grad_2, grad_2_auto, atol=3e-6), "CompositeSumCost implemented grad_2 not same as auto grad!"
    if not torch.allclose(grad_1, grad_1_auto):
        warnings.warn(f"CompositeSumCost implemented grad_1 deviates from auto grad value beyond torch.allclose default tolerances!!")
    if not torch.allclose(grad_2, grad_2_auto):
        warnings.warn(f"CompositeSumCost implemented grad_2 deviates from auto grad value beyond torch.allclose default tolerances!!")


def test_dimension_sum_cost():

    batch_size = 5
    x_dim_1 = (2,3)
    x_dim_2 = 4
    y_dim = (2,3)

    def mult(xs : Union[int, Iterable[int]]):
        if isinstance(xs, Iterable):
            prod = 1
            for i in xs:
                prod *= i
            return prod
        else:
            return xs

    class _TestCost(BaseCost):
        def __init__(self) -> None:
            super().__init__()
            _x_dim_1 = mult(x_dim_1)
            _x_dim_2 = mult(x_dim_2)
            _y_dim = mult(y_dim)
            self.A = torch.randn(_y_dim, _x_dim_1)
            self.B = torch.randn(_y_dim, _x_dim_2)
            self.x_dim_1 = _x_dim_1
            self.x_dim_2 = _x_dim_2

        def cost(self, x1, x2):
            batch_shape = x1.shape[:-len(x_dim_1)]
            x1 = x1.reshape(*batch_shape, self.x_dim_1, 1)
            x2 = x2.reshape(*batch_shape, self.x_dim_2, 1)
            y = self.A @ x1 + self.B @ x2
            return y.reshape(*batch_shape, *y_dim)

        def grad_w_cost(self, x1, x2):
            batch_shape = x1.shape[:-len(x_dim_1)]
            cost = self.cost(x1, x2)
            A = self.A.reshape(*y_dim, *x_dim_1).expand(*batch_shape, *y_dim, *x_dim_1)
            B = self.B.reshape(*y_dim, x_dim_2).expand(*batch_shape, *y_dim, x_dim_2)
            return A, B, cost

    dim_sum_cost = DimensionSumCost(_TestCost(), (2,1), (-2))
    x1 = torch.randn(batch_size, *x_dim_1, requires_grad=True)
    x2 = torch.randn(batch_size, x_dim_2, requires_grad= True)
    grad_1, grad_2, _ = dim_sum_cost.grad_w_cost(x1, x2)
    grad_1_auto, grad_2_auto = torch.vmap(torch.func.jacrev(dim_sum_cost.cost, argnums=(0,1)))(x1, x2)

    assert torch.allclose(grad_1, grad_1_auto, atol=3e-6), "CompositeSumCost implemented grad_1 not same as auto grad!"
    assert torch.allclose(grad_2, grad_2_auto, atol=3e-6), "CompositeSumCost implemented grad_2 not same as auto grad!"
    if not torch.allclose(grad_1, grad_1_auto):
        warnings.warn(f"CompositeSumCost implemented grad_1 deviates from auto grad value beyond torch.allclose default tolerances!!")
    if not torch.allclose(grad_2, grad_2_auto):
        warnings.warn(f"CompositeSumCost implemented grad_2 deviates from auto grad value beyond torch.allclose default tolerances!!")


def test_exponential_cost():
    batch_size = 3
    q_dim = 4
    sigma_q = 2.3
    sigma_exp = 4.3

    Q = torch.randn(q_dim, q_dim)
    exp_cost = ExponentialCost(QuadraticCost(Q, sigma_q), sigma=sigma_exp)
    x = torch.randn(batch_size, q_dim, requires_grad=True)
    grad, cost = exp_cost.grad_w_cost(x)
    grad_auto, = torch.autograd.grad(cost.sum(), x)

    assert torch.allclose(grad, grad_auto), "ExponentialCost implemented grad not same as auto grad!"


def test_pre_evaluated_cost():
    batch_size = 4
    q_dim = 5
    sigma_q = 0.3
    sigma_exp = 2.8

    Q = torch.randn(q_dim, q_dim)
    exp_cost = ExponentialCost(QuadraticCost(Q, sigma_q), sigma=sigma_exp)
    pre_eval_cost = PreEvaluatedCost(exp_cost)
    x = torch.randn(batch_size, q_dim, requires_grad=True)
    grad, cost = exp_cost.grad_w_cost(x)
    pre_eval_cost.pre_eval_grad_w_cost(x)
    grad_pre_eval, cost_pre_eval = pre_eval_cost.grad_w_cost(x)

    assert torch.allclose(cost, cost_pre_eval), "PreEvaluatedCost cost functions cost output not equivalent to stored cost output!"
    assert torch.allclose(grad, grad_pre_eval), "PreEvaluatedCost cost functions grad output not equivalent to stored cost output!"


def test_aabb_dist_2d_cost():
    batch_size = 4
    width = 1.6
    height = 2.4
    center = torch.randn(2)
    sigma = 1.07

    aabb_dist_cost = AABBDistance2DCost(width, height, center, sigma=sigma)
    x = center.clone().detach().requires_grad_(True) + \
        (width**2 + height**2)**0.5 * torch.randn(batch_size, 2, requires_grad=True)
    grad, cost = aabb_dist_cost.grad_w_cost(x)
    grad_auto, = torch.autograd.grad(cost.sum(), x)
    cost2 = aabb_dist_cost.cost(x)

    assert torch.allclose(grad, grad_auto), "AABBDistance2DCost implemented grad not same as auto grad!"
    assert torch.allclose(cost, cost2), "AABBDistance2DCost implemented cost from cost() and grad_w_cost() not the same!"


def test_circle_dist_2d_cost():
    batch_size = 8
    radius = 3.4
    center = torch.randn(2)
    sigma = 0.67

    circle_dist_cost = CircleDistance2DCost(radius, center, sigma=sigma)
    x = center.clone().detach().requires_grad_(True) + radius * torch.randn(batch_size, 2, requires_grad=True)
    grad, cost = circle_dist_cost.grad_w_cost(x)
    grad_auto, = torch.autograd.grad(cost.sum(), x)
    cost2 = circle_dist_cost.cost(x)

    assert torch.allclose(grad, grad_auto), "CircleDistance2DCost implemented grad not same as auto grad!"
    assert torch.allclose(cost, cost2), "CircleDistance2DCost implemented cost from cost() and grad_w_cost() not the same!"


def test_signed_dist_map_2d_cost():
    batch_size = 6
    width = 0.1
    height = 0.38
    aabb_center = torch.randn(2)
    radius = 0.46
    circle_center = torch.randn(2)

    signed_dist_map = SignedDistanceMap2DCost(())
    signed_dist_map.add_aabb(width, height, aabb_center)
    signed_dist_map.add_circle(radius,circle_center)
    x = torch.randn(batch_size, 2, requires_grad=True)
    grad, cost = signed_dist_map.grad_w_cost(x)
    grad_auto, = torch.autograd.grad(cost.sum(), x)
    cost2 = signed_dist_map.cost(x)

    assert torch.allclose(grad, grad_auto), "SignedDistMap2DCost implemented grad not same as auto grad!"
    assert torch.allclose(cost, cost2), "SignedDistMap2DCost implemented cost from cost() and grad_w_cost() not the same!"


def test_exponential_sum_obstacle_cost():
    num_of_diff = 3
    batch_size = 5
    width = 0.2
    height = 0.4
    aabb_center = torch.randn(2)
    radius = 0.32
    circle_center = torch.randn(2)
    sigma_obs_in = 2
    sigma_obs_out = 50
    sigma = 0.67

    signed_dist_map = SignedDistanceMap2DCost(())
    signed_dist_map.add_aabb(width, height, aabb_center)
    signed_dist_map.add_circle(radius,circle_center)
    exp_sum_cost = ExponentialSumObstacleCost(signed_dist_map, sigma_obs_in, sigma_obs_out, sigma=sigma)
    x = torch.randn(batch_size, 2 * num_of_diff, requires_grad=True)
    grad, cost = exp_sum_cost.grad_w_cost(x)
    grad_auto, = torch.autograd.grad(cost.sum(), x)
    cost2 = exp_sum_cost.cost(x)

    assert torch.allclose(grad, grad_auto), "ExponentialSumObstacleCost implemented grad not same as auto grad!"
    assert torch.allclose(cost, cost2), "ExponentialSumObstacleCost implemented cost from cost() and grad_w_cost() not the same!"


def test_k_bending_obstacle_cost():
    num_of_diff = 3
    batch_size = 6
    aabb_width = 0.13
    aabb_height = 0.52
    aabb_center = torch.randn(2)
    circle_radius = 0.13
    circle_center = torch.randn(2)
    T = 14
    critical_radius = 0.4
    k_bend = 0.2
    sigma = 0.67
    sigma_T = torch.linspace(90, 20, T)

    signed_dist_map = SignedDistanceMap2DCost(())
    signed_dist_map.add_aabb(aabb_width, aabb_height, aabb_center)
    signed_dist_map.add_circle(circle_radius,circle_center)
    k_bending_obs_cost = KBendingObstacleCost(signed_dist_map, critical_radius, k_bend, sigma_T, sigma)
    x = torch.randn(batch_size, T, 2 * num_of_diff, requires_grad=True)
    grad, cost = k_bending_obs_cost.grad_w_cost(x)
    grad_auto = torch.vmap(torch.func.jacrev(k_bending_obs_cost.cost))(x)
    cost2 = k_bending_obs_cost.cost(x)

    assert torch.allclose(grad, grad_auto), "KBendingObstacleCost implemented grad not same as auto grad!"
    assert torch.allclose(cost, cost2), "KBendingObstacleCost implemented cost from cost() and grad_w_cost() not the same!"


def test_running_delta_cost():
    batch_size = 5
    base_dim = 3
    num_diff = 4
    T = 20

    Qs = [torch.randn(base_dim, base_dim) for _ in range(num_diff)]
    x_bars = [torch.randn(base_dim) for _ in range(num_diff)]
    sigma = 2.8

    running_delta_cost = RunningDeltaCost(Qs, x_bars, sigma=sigma)
    x = torch.randn(batch_size, T, base_dim * num_diff, requires_grad=True)
    grad, cost = running_delta_cost.grad_w_cost(x)
    grad_auto, = torch.autograd.grad(cost.sum(), x)

    assert torch.allclose(grad, grad_auto, atol=3e-6), "RunningDeltaCost implemented grad not same as auto grad!"
    if not torch.allclose(grad, grad_auto):
        warnings.warn(f"RunningDeltaCost implemented grad deviates from auto grad value beyond torch.allclose default tolerances!!")


def test_terminal_delta_cost():
    batch_size = 6
    base_dim = 2
    num_diff = 5
    T = 32

    Qs = [torch.randn(base_dim, base_dim) for _ in range(num_diff)]
    x_bars = [torch.randn(base_dim) for _ in range(num_diff)]
    sigma = 2.8

    terminal_delta_cost = TerminalDeltaCost(Qs, x_bars, sigma=sigma)
    x = torch.randn(batch_size, T, base_dim * num_diff, requires_grad=True)
    grad, cost = terminal_delta_cost.grad_w_cost(x)
    grad_auto, = torch.autograd.grad(cost.sum(), x)

    assert torch.allclose(grad, grad_auto, atol=3e-6), "TerminalDeltaCost implemented grad not same as auto grad!"
    if not torch.allclose(grad, grad_auto):
        warnings.warn(f"TerminalDeltaCost implemented grad deviates from auto grad value beyond torch.allclose default tolerances!!")


def test_running_collision_cost():
    batch_size = 4
    base_dim = 2
    num_diff = 5
    pos_dim = 2
    T = 3
    radius = 6.7
    k_bend = 1 # 0.3
    sigma_T = torch.linspace(100, 10, T)
    sigma = 4.3

    runing_collision_cost = RunningCollisionCost(pos_dim, radius, k_bend, sigma_T, sigma=sigma)
    x1 = torch.randn(batch_size, T, base_dim * num_diff, requires_grad=True)
    x2 = x1.clone().detach() + radius * torch.randn(batch_size, T, base_dim * num_diff, requires_grad=True)
    grad_1, grad_2, cost = runing_collision_cost.grad_w_cost(x1, x2)
    grad_1_auto, grad_2_auto = torch.autograd.grad(cost.sum(), (x1, x2))
    cost2 = runing_collision_cost.cost(x1,x2)

    assert torch.allclose(grad_1, grad_1_auto, atol=3e-6), "RunningCollisionCost implemented grad_1 not same as auto grad!"
    assert torch.allclose(grad_2, grad_2_auto, atol=3e-6), "RunningCollisionCost implemented grad_2 not same as auto grad!"
    if not torch.allclose(grad_1, grad_1_auto):
        warnings.warn(f"RunningCollisionCost implemented grad_1 deviates from auto grad value beyond torch.allclose default tolerances!!")
    if not torch.allclose(grad_2, grad_2_auto):
        warnings.warn(f"RunningCollisionCost implemented grad_2 deviates from auto grad value beyond torch.allclose default tolerances!!")
    assert torch.allclose(cost, cost2), "RunningCollisionCost implemented cost from cost() and grad_w_cost() not the same!"


def test_running_cross_collision_cost():

    batch_size = 4
    num_particles_1 = 5
    num_particles_2 = 6
    base_dim = 2
    num_diff = 3
    pos_dim = 2
    T = 3
    radius = 0.59
    k_bend = 0.4
    sigma_T = torch.linspace(90, 20, T)
    sigma = 3.82

    runing_cross_collision_cost = RunningCrossCollisionCost(pos_dim, radius, k_bend, sigma_T, sigma=sigma)
    x1 = torch.randn(batch_size, num_particles_1, T, base_dim * num_diff, requires_grad=True)
    x2 = torch.randn(batch_size, num_particles_2, T, base_dim * num_diff, requires_grad=True)
    grad_1, grad_2, cost = runing_cross_collision_cost.grad_w_cost(x1, x2)
    cost2 = runing_cross_collision_cost.cost(x1,x2)


    grad_x1_auto, grad_x2_auto = torch.vmap(
        torch.func.jacrev(runing_cross_collision_cost.cost, argnums=(0,1))
        )(x1, x2)
    grad_x1_auto = grad_x1_auto.sum(dim=-3) # remove trivial dimension
                                            # (trajectories in x1 doesnt affect collision calculation of
                                            # other x1 trajectories with x2 trajectories)
    grad_x2_auto = grad_x2_auto.sum(dim=-3) # remove trivial dimension
                                            # (trajectories in x2 doesnt affect collision calculation of
                                            # other x2 trajectories with x1 trajectories)

    assert torch.allclose(grad_1, grad_x1_auto, atol=3e-6), "RunningCrossCollisionCost implemented grad_1 not same as auto grad!"
    assert torch.allclose(grad_2, grad_x2_auto, atol=3e-6), "RunningCrossCollisionCost implemented grad_2 not same as auto grad!"
    if not torch.allclose(grad_1, grad_x1_auto):
        warnings.warn(f"RunningCrossCollisionCost implemented grad_1 deviates from auto grad value beyond torch.allclose default tolerances!!")
    if not torch.allclose(grad_2, grad_x2_auto):
        warnings.warn(f"RunningCrossCollisionCost implemented grad_2 deviates from auto grad value beyond torch.allclose default tolerances!!")
    assert torch.allclose(cost, cost2), "RunningCrossCollisionCost implemented cost from cost() and grad_w_cost() not the same!"