"""
Run tests to ensure that LinearPointRobot implementation had been replicated
"""

import torch

from multi_robot_svbp.sim.robot import LinearDynamicsModel, StackedTrajectoryForm

"""
Old implementation
"""

class LinearPointRobot_old(object):
    def __init__(self, x0, x_dim=4, u_dim=2, pos_lims=None, vel_lims=None, dt=0.1, horizon=1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}):
        self.x = x0.clone().detach() if torch.is_tensor(x0) else torch.tensor(x0)
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.pos_lims = pos_lims
        self.vel_lims = vel_lims
        self.dt = dt
        self.tensor_kwargs = tensor_kwargs

        self.x = self.x.to(**tensor_kwargs)

        self.A = torch.eye(self.x_dim, **tensor_kwargs)
        self.A[0, 2] = self.dt
        self.A[1, 3] = self.dt
        self.B = torch.cat((0.5 * self.dt**2 * torch.eye(self.u_dim, **tensor_kwargs),
                            self.dt * torch.eye(self.u_dim, **tensor_kwargs)), dim=0)

        # Batch params
        self.T = horizon
        self.batch_A, self.batch_B = self._create_batch_matrices(horizon)

    def reset(self, x0):
        self.x = x0.clone().detach() if torch.is_tensor(x0) else torch.tensor(x0)
        self.x = self.x.to(**self.tensor_kwargs)

    def step(self, u):
        self.x = self._linear_model(self.x, u)

    def set_horizon(self, horizon):
        self.T = horizon
        self.batch_A, self.batch_B = self._create_batch_matrices(horizon)

    def rollout(self, u, state=None):
        assert u.size(-1) == self.u_dim, "Control input has wrong number of dimensions, {}".format(u.size(-1))
        assert u.ndim == 3 or u.ndim == 2, "Control input must be shape (N, T, D) or (T, D)."

        if state is not None:
            self.reset(state)

        N = u.size(0) if u.ndim == 3 else 1
        T = u.size(-2)
        bA, bB = self.batch_A, self.batch_B

        x_0 = self.x.clone().detach().view(1, self.x_dim)
        u = u.view(N, T * self.u_dim)
        traj = torch.matmul(x_0, bA.T) + torch.matmul(u, bB.T)
        traj = torch.cat([x_0.view(-1).repeat(N, 1, 1), traj.reshape(N, T, self.x_dim)], dim=1)
        if N == 1:
            traj = traj.reshape(T + 1, self.x_dim)

        # Slow way.
        # traj = [self.x.clone().detach()]
        # for i in range(T):
        #     traj.append(self._linear_model(traj[-1], u[i]))
        # traj = torch.stack(traj)

        return traj

    def _linear_model(self, x, u):
        return torch.matmul(self.A, x) + torch.matmul(self.B, u)

    def _create_batch_matrices(self, T):
        bA = torch.cat([torch.linalg.matrix_power(self.A, i + 1) for i in range(T)], dim=0)
        bB = torch.eye(self.x_dim * T, **self.tensor_kwargs)
        for i in range(0, T - 1):
            bB[i * self.x_dim:(i + 1) * self.x_dim,
               (i + 1) * self.x_dim:] = bA.T[:, :(T - i - 1) * self.x_dim]
        bB = bB.T.mm(torch.block_diag(*[self.B for i in range(T)]))

        return bA, bB

"""
Tests
"""

def test_linear_point_robot_replicated():
    batch_size = 6
    T = 3
    dt = 0.2
    pos_dim = 2
    x_dim = 2 * pos_dim
    u_dim = 1 * pos_dim

    u = torch.randn(batch_size, T, u_dim)
    x0 = torch.randn(x_dim)

    robot = LinearPointRobot_old(x0, dt=dt, horizon=T)
    old_traj = torch.cat((robot.rollout(u)[...,1:,:], u), dim =-1)
    old_u_grad = robot.batch_B.view(T, x_dim, T, u_dim) # reshaped for easy comparison
                                                        # note: batch_B only provides gradients up for outputs [x_1 ... x_N] (ie excludes x_0) and also excludes all Us
                                                        # note: since linear, batch_B applies to ALL batches (therefore batch dimension was eliminated)

    linear_dyn_model = LinearDynamicsModel(robot.A, robot.B, T=T)
    stacked_traj_form = StackedTrajectoryForm(linear_dyn_model) # includes t=0
    new_u_grad, _, new_traj = stacked_traj_form.rollout_w_grad(u, x0)

    assert torch.allclose(old_traj, new_traj), "LinearDynamicsModel trajectory not equivalent to LinearRobot output!"
    assert torch.allclose(old_u_grad, new_u_grad[...,:,:x_dim,:,:]), "LinearDynamicsModel trajectory not equivalent to LinearRobot output!"