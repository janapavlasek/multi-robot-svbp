"""
Tests to check that collision avoid sbp factors are replicated
"""

import torch

from multi_robot_svbp.costs.base_costs import CompositeSumCost
from multi_robot_svbp.costs.trajectory_costs import RunningDeltaCost, TerminalDeltaCost, RunningCrossCollisionCost
from multi_robot_svbp.costs.obstacle_costs import SignedDistanceMap2DCost, ExponentialSumObstacleCost
from multi_robot_svbp.sim.robot import LinearDynamicsModel, StackedTrajectoryForm

from unit_tests.replicate_diff_map_test import DiffMap_old
from unit_tests.replicate_linear_pt_robot_test import LinearPointRobot_old

from torch_bp.util.misc import euclidean_distance

import warnings

"""
Tests
"""

def test_unary_robot_factor_replicated():

    def _log_likelihood_old(x, goal, horizon, dim, dmap, sigma_obs, c_obs, c_pos, c_vel, c_u, c_term):
        """
        Replicated function here in case UnaryRobotFactor is changed in the future
        """
        K = x.size(dim=0)
        x = x.view(-1, horizon, dim * 3)

        pos, vel, u = x[:, :, :dim], x[:, :, dim:2 * dim], x[:, :, 2 * dim:]

        # Obstacle cost.
        sdf = dmap.eval_sdf(pos.view(K * horizon, dim))
        val = sigma_obs * sdf.clamp(max=0) + 2 * sdf.clamp(min=0)
        # val = self.sigma_obs * sdf
        cost = c_obs * torch.exp(val).view(K, horizon).sum(1)

        # Quadratic cost.
        if goal is not None:
            pos = pos - goal

        pos_cost = c_pos * (pos * pos).sum(dim=-1)
        vel_cost = c_vel * (vel * vel).sum(dim=-1)
        u_cost = c_u * (u * u).sum(dim=-1)

        cost += pos_cost.sum(-1) + vel_cost.sum(-1) + u_cost.sum(-1)

        # Terminal goal cost.
        term_pos, term_vel = pos[:, -1, :], vel[:, -1, :]
        term_cost = c_term * (term_pos * term_pos).sum(dim=-1)
        cost += term_cost

        return -cost

    def _grad_log_likelihood_old(x, drollout, goal, horizon, dim, dmap, sigma_obs, c_obs, c_pos, c_vel, c_u, c_term):
        """
        Replicated function here in case UnaryRobotFactor is changed in the future
        """
        K = x.size(0)

        # Use Autograd to get gradients. TODO: Do these manually.
        x = x.detach().requires_grad_(True)
        log_px = _log_likelihood_old(x, goal, horizon, dim, dmap, sigma_obs, c_obs, c_pos, c_vel, c_u, c_term)
        grad_log_px, = torch.autograd.grad(log_px.sum(), x)

        grad_log_px = grad_log_px.view(K, horizon, dim * 3)
        # Grab the gradient of the costs for the state: dC / dx
        grad_traj = grad_log_px[:, :, :2 * dim].reshape(K, horizon * dim * 2)
        # Add the gradient of the costs related to the position and velocity w.r.t. acceleration.
        # Uses the chain rule: dC / du = (dC / dx) * (df(u) / du) where x = f(u).
        grad_log_px[:, :, -dim:] += grad_traj.matmul(drollout).view(K, horizon, dim)
        # Zero out gradient for state.
        grad_log_px[:, :, :-dim] = 0

        return grad_log_px.view(K, horizon * dim * 3), log_px


    batch_size = 6
    T = 3
    dt = 0.2
    pos_dim = 2
    x_dim = 2 * pos_dim
    u_dim = 1 * pos_dim
    c_pos=0.1
    c_vel=0.25
    c_u=0.2
    c_term=8.
    c_obs=10000
    sigma_obs=50

    aabb_width = 0.12
    aabb_height = 0.34
    aabb_center = torch.randn(2)
    circle_radius = 0.23
    circle_center = torch.randn(2)


    u = torch.randn(batch_size, T, u_dim)
    x0 = torch.randn(x_dim)
    goal = torch.randn(pos_dim)

    robot = LinearPointRobot_old(x0, dt=dt, horizon=T)
    traj_old = torch.cat((robot.rollout(u)[...,1:,:], u), dim =-1)

    diffmap = DiffMap_old(width=10.,height=10.,origin=[-5,-5])
    diffmap.add_rectangle(aabb_width, aabb_height, aabb_center)
    diffmap.add_circle(circle_radius,circle_center)
    old_grad, old_cost = _grad_log_likelihood_old(x=traj_old, drollout=robot.batch_B, goal=goal, horizon=T, dim=pos_dim, dmap=diffmap, sigma_obs=sigma_obs, c_obs=c_obs, c_pos=c_pos, c_vel=c_vel, c_u=c_u, c_term=c_term)
    old_grad = old_grad.view(batch_size, T, x_dim+u_dim) # reshape for easy comparison

    linear_dyn_model = LinearDynamicsModel(robot.A, robot.B, T=T)
    stacked_traj_form = StackedTrajectoryForm(linear_dyn_model)
    traj_u_grad, _, traj = stacked_traj_form.rollout_w_grad(u, x0)
    diff_map = SignedDistanceMap2DCost(dist_2d_costs=())
    diff_map.add_aabb(aabb_width, aabb_height, aabb_center)
    diff_map.add_circle(circle_radius,circle_center)
    obs_cost = ExponentialSumObstacleCost(signed_2d_map=diff_map, sigma_obs_out=sigma_obs, sigma_obs_in=2, sigma=c_obs)
    running_cost_Qs = (c_pos * torch.eye(pos_dim), c_vel * torch.eye(pos_dim), c_u * torch.eye(pos_dim))
    running_cost_x_bars = (goal, torch.zeros_like(goal), torch.zeros_like(goal))
    running_cost = RunningDeltaCost(Qs=running_cost_Qs, x_bars=running_cost_x_bars)
    terminal_cost_Qs = (c_term * torch.eye(pos_dim), 0 * torch.eye(pos_dim), 0 * torch.eye(pos_dim))
    terminal_cost_x_bars = (goal, torch.zeros_like(goal), torch.zeros_like(goal))
    terminal_cost = TerminalDeltaCost(Qs=terminal_cost_Qs, x_bars=terminal_cost_x_bars)
    combined_cost = CompositeSumCost(costs=(obs_cost, running_cost, terminal_cost), sigma=-1)
    traj_cost_grad, new_cost = combined_cost.grad_w_cost(traj)
    # dc/d(U) = dc/d(X_bar) @ d(X_bar)/d(U)
    new_grad = traj_cost_grad.view(batch_size, 1, T*(x_dim+u_dim)) @ traj_u_grad.view(batch_size, T*(x_dim+u_dim), T*u_dim)
    new_grad = new_grad.view(batch_size, T, u_dim) # reshape for easy comparison

    assert torch.allclose(old_cost, new_cost, atol=3e-6), "New Composite cost functions cost output not equivalent to UnaryRobotFactor output!"
    assert torch.allclose(old_grad[...,-u_dim:], new_grad, atol=3e-6), "New Composite cost functions grad output not equivalent to UnaryRobotFactor output!"
    if not torch.allclose(old_cost, new_cost):
        warnings.warn(f"New cost function's cost deviates from UnaryRobotFactor value beyond torch.allclose default tolerances!!")
    if not torch.allclose(old_grad[...,-u_dim:], new_grad):
        warnings.warn(f"New cost function's grad deviates from UnaryRobotFactor value beyond torch.allclose default tolerances!!")

def test_trajectory_collision_factor_replicated():

    def _adjust_ccoll_for_ccollend(c_coll, c_coll_end, horizon):
        """
        Replicated function here in case TrajectoryCollisionFactor is changed in the future
        """
        if c_coll_end is not None:
            return torch.linspace(c_coll, c_coll_end, horizon)
        else:
            return c_coll

    def _log_likelihood_old(x_s, x_t, horizon, dim, k_bend, r, c_coll):
        """
        Replicated function here in case TrajectoryCollisionFactor is changed in the future
        """
        x_s = x_s.view(-1, horizon, dim * 3)
        x_t = x_t.view(-1, horizon, dim * 3)
        # Size (Ns, Nt, T)
        pos_diff = 1 - euclidean_distance(x_s[:, :, :2], x_t[:, :, :2], squared=False)**k_bend / r**k_bend
        pos_diff = pos_diff.clamp(min=0)  # No impact from anything larger than the radius away.
        cost = (pos_diff * c_coll).sum(-1)  # Sum over the trajectory.
        return -cost

    def _grad_log_likelihood_old(x_s, x_t, drollout, horizon, dim, k_bend, r, c_coll):
        """
        Replicated function here in case TrajectoryCollisionFactor is changed in the future
        """
        Ns = x_s.size(0)
        Nt = x_t.size(0)

        jac_i = torch.vmap(
            torch.func.jacrev(_log_likelihood_old, argnums=1), in_dims=(None, 0, None, None, None, None, None))
        dpair_1 = jac_i(x_t, x_s, horizon, dim, k_bend, r, c_coll).view(Nt, Ns, horizon, dim * 3)
        dpair_2 = jac_i(x_s, x_t, horizon, dim, k_bend, r, c_coll).view(Ns, Nt, horizon, dim * 3)

        d_traj_1 = dpair_1[:, :, :, :-2].reshape(Nt, Ns, horizon * dim * 2)
        d_traj_2 = dpair_2[:, :, :, :-2].reshape(Ns, Nt, horizon * dim * 2)

        # Zero out any impact from the position or velocity. Only the acceleration command has a gradient.
        dpair_1_out = torch.zeros_like(dpair_1)
        dpair_2_out = torch.zeros_like(dpair_2)
        # This gives d log_px(traj) / dtraj * d f(u) / du
        dpair_1_out[:, :, :, -2:] = d_traj_1.matmul(drollout).view(Nt, Ns, horizon, dim)
        dpair_2_out[:, :, :, -2:] = d_traj_2.matmul(drollout).view(Ns, Nt, horizon, dim)

        # TODO: Return this in the Jacobian.
        log_px = _log_likelihood_old(x_s, x_t, horizon, dim, k_bend, r, c_coll)

        # return (dpair_1_out.view(Ns, Nt, horizon * dim * 3),
        #         dpair_2_out.view(Ns, Nt, horizon * dim * 3),
        #         log_px)
        return (dpair_1_out.view(Ns, Nt, horizon * dim * 3),
                dpair_2_out.view(Nt, Ns, horizon * dim * 3).transpose(0,1),
                log_px) # confirm with Jana if new implementation is more accurate

    num_particles_s = 7
    num_particles_t = 8
    T = 5
    dt = 0.2
    pos_dim = 2
    x_dim = 2 * pos_dim
    u_dim = 1 * pos_dim
    k_bend = 0.3
    r = 2.0
    c_coll = 100.
    c_coll_end = 10

    c_coll = -_adjust_ccoll_for_ccollend(c_coll, c_coll_end, T)

    u_s = torch.randn(num_particles_s, T, u_dim)
    x0_s = torch.randn(x_dim)
    u_t = torch.randn(num_particles_t, T, u_dim)
    x0_t = x0_s

    robot_s = LinearPointRobot_old(x0_s, dt=dt, horizon=T)
    traj_s_old = torch.cat((robot_s.rollout(u_s)[...,1:,:], u_s), dim =-1)
    robot_t = LinearPointRobot_old(x0_t, dt=dt, horizon=T)
    traj_t_old = torch.cat((robot_t.rollout(u_t)[...,1:,:], u_t), dim =-1)

    old_grad_xs, old_grad_xt, old_cost = _grad_log_likelihood_old(
        x_s=traj_s_old, x_t=traj_t_old, drollout=robot_s.batch_B, horizon=T, dim=pos_dim,
        k_bend=k_bend, r=r, c_coll=c_coll)
    old_grad_xs = old_grad_xs.view(num_particles_s, num_particles_t, T, x_dim+u_dim) # reshape for easy comparison
    old_grad_xt = old_grad_xt.view(num_particles_s, num_particles_t, T, x_dim+u_dim) # reshape for easy comparison

    linear_dyn_model_s = LinearDynamicsModel(robot_s.A, robot_s.B, T=T)
    stacked_traj_form_s = StackedTrajectoryForm(linear_dyn_model_s)
    traj_s_u_grad, _, traj_s = stacked_traj_form_s.rollout_w_grad(u_s, x0_s)
    linear_dyn_model_t = LinearDynamicsModel(robot_t.A, robot_t.B, T=T)
    stacked_traj_form_t = StackedTrajectoryForm(linear_dyn_model_t)
    traj_t_u_grad, _, traj_t = stacked_traj_form_t.rollout_w_grad(u_t, x0_t)

    new_cost_fn = RunningCrossCollisionCost(
        pos_dim=pos_dim, radius=r, k_bend=k_bend, sigma_T=c_coll, sigma=-1)
    traj_s_cost_grad, traj_t_cost_grad, new_cost = new_cost_fn.grad_w_cost(traj_s, traj_t)
    # dc/d(U_s) = dc/d(X_bar_s) @ d(X_bar_s)/d(U_s)
    new_grad_xs = traj_s_cost_grad.view(num_particles_s, num_particles_t, 1, T*(x_dim+u_dim)) @ \
        traj_s_u_grad.view(num_particles_s, 1, T*(x_dim+u_dim), T*u_dim)
    new_grad_xs = new_grad_xs.view(num_particles_s, num_particles_t, T, u_dim) # reshape for easy comparison
    # dc/d(U_t) = dc/d(X_bar_t) @ d(X_bar_t)/d(U_t)
    new_grad_xt = traj_t_cost_grad.view(num_particles_s, num_particles_t, 1, T*(x_dim+u_dim)) @ \
        traj_t_u_grad.view(1, num_particles_t, T*(x_dim+u_dim), T*u_dim)
    new_grad_xt = new_grad_xt.view(num_particles_s, num_particles_t, T, u_dim) # reshape for easy comparison

    assert torch.allclose(old_cost, new_cost, atol=3e-6), "New cost functions cost output not equivalent to TrajectoryCollisionFactor output!"
    assert torch.allclose(old_grad_xs[...,-u_dim:], new_grad_xs, atol=3e-6), "New cost functions grad_xs output not equivalent to TrajectoryCollisionFactor output!"
    assert torch.allclose(old_grad_xt[...,-u_dim:], new_grad_xt, atol=3e-6), "New cost functions grad_xt output not equivalent to TrajectoryCollisionFactor output!"
    if not torch.allclose(old_cost, new_cost):
        warnings.warn(f"New cost functions cost deviates from TrajectoryCollisionFactor value beyond torch.allclose default tolerances!!")
    if not torch.allclose(old_grad_xs[...,-u_dim:], new_grad_xs):
        warnings.warn(f"New cost functions grad_xs deviates from TrajectoryCollisionFactor value beyond torch.allclose default tolerances!!")
    if not torch.allclose(old_grad_xt[...,-u_dim:], new_grad_xt):
        warnings.warn(f"New cost functions grad_xt deviates from TrajectoryCollisionFactor value beyond torch.allclose default tolerances!!")