import torch
import torch_bp.bp as bp
from torch_bp.util.misc import euclidean_distance

from ..sim.robot import LinearPointRobotModel


class AnchorFactor(bp.factors.UnaryFactor):
    def __init__(self, state, c=1, dim=2):
        super().__init__()
        self.anchor_state = state
        self.c = c
        self.dim = dim

    def log_likelihood(self, x):
        diff = x - self.anchor_state
        diff = diff * diff

        cost = self.c * diff.sum(dim=-1)
        return -cost


class QuadraticFactor(bp.factors.UnaryFactor):
    def __init__(self, state, c_pos=0., c_vel=0.25, c_u=0.2, dim=2, horizon=1, dt=0.1, goal=None,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float}):
        super().__init__()
        self.robot = LinearPointRobotModel(dt=dt, horizon=horizon, tensor_kwargs=tensor_kwargs)
        self.state = state
        self.c_pos = c_pos
        self.c_vel = c_vel
        self.c_u = c_u
        self.dim = dim
        self.horizon = horizon
        self.goal = goal

    def set_state(self, state):
        self.state = state

    def log_likelihood(self, x):
        traj = self.robot.rollout(x.view(-1, self.horizon, self.robot.u_dim), self.state)
        pos, vel = traj[:, :, :self.dim], traj[:, :, self.dim:2 * self.dim]
        u = x.view(-1, self.horizon, self.dim)

        if self.goal is not None:
            pos = pos - self.goal

        pos_cost = self.c_pos * (pos * pos).sum(dim=-1)
        vel_cost = self.c_vel * (vel * vel).sum(dim=-1)
        u_cost = self.c_u * (u * u).sum(dim=-1)

        cost = pos_cost.sum(-1) + vel_cost.sum(-1) + u_cost.sum(-1)
        return -cost


class FlockFactor(bp.factors.PairwiseFactor):
    def __init__(self, state1, state2, c_vel=0.5, dt=0.1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float}, **kwargs):
        super().__init__(**kwargs)
        self.state1 = state1.detach().clone() if torch.is_tensor(state1) else torch.as_tensor(state1)
        self.state2 = state2.detach().clone() if torch.is_tensor(state2) else torch.as_tensor(state2)
        self.state1, self.state2 = self.state1.to(**tensor_kwargs), self.state2.to(**tensor_kwargs)
        self.c_vel = c_vel

        self.A = torch.eye(4, **tensor_kwargs)
        self.A[0, 2] = dt
        self.A[1, 3] = dt
        self.B = torch.cat((0.5 * dt**2 * torch.eye(2), dt * torch.eye(2)), dim=0).to(**tensor_kwargs)

    def log_likelihood(self, x_s, x_t):
        x1_next = torch.matmul(self.state1, self.A.T) + torch.matmul(x_s, self.B.T)
        x2_next = torch.matmul(self.state2, self.A.T) + torch.matmul(x_t, self.B.T)
        vel_diff = euclidean_distance(x1_next[:, 2:4], x2_next[:, 2:4])
        return -self.c_vel * vel_diff


class TrajectoryFlockFactor(bp.factors.PairwiseFactor):
    def __init__(self, state1, state2, c_vel=0.5, dt=0.1, horizon=1,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float}, **kwargs):
        super().__init__(**kwargs)
        self.robot = LinearPointRobotModel(dt=dt, horizon=horizon, tensor_kwargs=tensor_kwargs)
        self.horizon = horizon
        self.c_vel = c_vel
        self.state1, self.state2 = state1, state2

    def log_likelihood(self, x_s, x_t):
        traj_1 = self.robot.rollout(x_s.view(-1, self.horizon, self.robot.u_dim), self.state1)
        traj_2 = self.robot.rollout(x_t.view(-1, self.horizon, self.robot.u_dim), self.state2)

        vel_diff = euclidean_distance(traj_1[..., 2:4], traj_2[..., 2:4])  # Size (Ns, Nt, T)
        vel_diff = vel_diff.sum(-1)
        return -self.c_vel * vel_diff
