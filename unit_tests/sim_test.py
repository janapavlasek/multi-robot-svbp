"""
Test codes for testing sim related modules
"""

from multi_robot_svbp.sim.robot import *

def test_linear_dynamics_model():
    batch_size = 8
    x_dim = 6
    u_dim = 3
    T = 5

    A = torch.randn(x_dim, x_dim)
    B = torch.randn(x_dim, u_dim)
    linear_dyn_model = LinearDynamicsModel(A, B, T)
    u = torch.randn(batch_size, T, u_dim, requires_grad=True)
    x0 = torch.randn(batch_size, x_dim, requires_grad=True)
    grad_u, grad_x0, traj = linear_dyn_model.rollout_w_grad(u, x0)
    grad_u_auto, grad_x0_auto = torch.vmap(torch.func.jacrev(linear_dyn_model.rollout, argnums=(0,1)))(u, x0)
    traj2 = linear_dyn_model.rollout(u, x0)

    assert torch.allclose(grad_u, grad_u_auto), "LinearDynamicsModel implemented grad_u not same as auto grad!"
    assert torch.allclose(grad_x0, grad_x0_auto), "LinearDynamicsModel implemented grad_x0 not same as auto grad!"
    assert torch.allclose(traj, traj2), "LinearDynamicsModel implemented traj from rollout() and rollout_w_grad() not the same!"


def test_stacked_trajectory_form():
    batch_size = 7
    x_dim = 6
    u_dim = 2
    T = 4

    A = torch.randn(x_dim, x_dim)
    B = torch.randn(x_dim, u_dim)
    linear_dyn_model = LinearDynamicsModel(A, B, T)
    stacked_traj_form = StackedTrajectoryForm(linear_dyn_model)
    u = torch.randn(batch_size, T, u_dim, requires_grad=True)
    x0 = torch.randn(batch_size, x_dim, requires_grad=True)
    grad_u, grad_x0, traj = stacked_traj_form.rollout_w_grad(u, x0)
    grad_u_auto, grad_x0_auto = torch.vmap(torch.func.jacrev(stacked_traj_form.rollout, argnums=(0,1)))(u, x0)
    traj2 = stacked_traj_form.rollout(u, x0)

    assert torch.allclose(grad_u, grad_u_auto), "StackedTrajectoryForm implemented grad_u not same as auto grad!"
    assert torch.allclose(grad_x0, grad_x0_auto), "StackedTrajectoryForm implemented grad_x0 not same as auto grad!"
    assert torch.allclose(traj, traj2), "StackedTrajectoryForm implemented traj from rollout() and rollout_w_grad() not the same!"
