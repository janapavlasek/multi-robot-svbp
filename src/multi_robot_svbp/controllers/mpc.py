import os
import torch
import numpy as np

from torch_bp.inference.svgd import SVGD
from torch_bp.inference.kernels import RBFMedianKernel

from multi_robot_svbp.sim.robot import LinearPointRobotModel
from multi_robot_svbp.factors.trajectory_factors import UnaryRobotTrajectoryFactor


class SteinMPC(object):
    def __init__(self, costs, num_particles, dt=0.1, horizon=1, dim=2,
                 sample_mode="mean", hotstart=True, init_cov=0.5,
                 shift_mode="shift",  # Shift the trajectories. Options: "shift", "best", "reset"
                 optim_params={"lr": 0.05}, tensor_kwargs={"device": torch.device("cpu"), "dtype": torch.float32}):
        self.num_particles = num_particles
        self.horizon = horizon
        self.dim = dim
        self.d_action = 3 * dim
        self.init_cov = init_cov
        self.sample_mode = sample_mode
        self.hotstart = hotstart
        self.shift_mode = shift_mode
        self.optim_params = optim_params
        self.tensor_kwargs = tensor_kwargs

        self.num_steps = 0

        # Robot state.
        self.state = torch.zeros(dim * 2, **self.tensor_kwargs)

        p_size = (self.num_particles, self.horizon, self.dim)
        init_u = torch.normal(0, self.init_cov, p_size, **self.tensor_kwargs)

        # NOTE: only allows linear models for now, for non-linear models we have to change the control flow!!
        self.robot_model = LinearPointRobotModel(dim, dt=dt, horizon=self.horizon, tensor_kwargs=tensor_kwargs)
        state_grads, _, init_particles = self.robot_model.rollout_w_grad(init_u, self.state[None, :])

        # The log likelihood function is just the unary function for a graph with one node.
        self.log_likelihood = UnaryRobotTrajectoryFactor(costs, horizon=self.horizon, dim=self.dim,
                                                         traj_grads_U=state_grads[0], tensor_kwargs=tensor_kwargs)

        gamma = 1. / np.sqrt(2 * self.horizon * self.dim)
        rbf_kernel = RBFMedianKernel(gamma=gamma)

        self.svgd = SVGD(init_particles, self.log_likelihood, rbf_kernel, grad_log_px=self.grad_log_likelihood)
        self.init_optimizer()

    def grad_log_likelihood(self, x):
        grad_px, _ = self.log_likelihood.grad_log_likelihood(x)
        return grad_px

    def init_optimizer(self):
        self.optim = torch.optim.Adam([self.svgd.optim_parameters()], **self.optim_params)

    def action_particles(self):
        return self.svgd.particles().view(self.num_particles, self.horizon, self.d_action)[..., -self.dim:]

    def rollout(self, state=None, actions=None):
        if state is None:
            state = self.state
        if actions is None:
            actions = self.action_particles()
        return self.robot_model.rollout(actions, state)

    def _get_action_seq(self, mode="mean", return_idx=False):
        particles = self.svgd.particles()
        action, idx = None, None
        if mode == "mean":
            action = torch.mean(particles, dim=0)
        elif mode == "best":
            weights = self.svgd.calc_weights()
            idx = weights.argmax().item()
            action = particles[weights.argmax(), :]
        elif mode == "sample":
            idx = torch.randint(0, self.num_particles, (1,)).item()
            action = particles[idx, :]
        else:
            raise ValueError("Unidentified sampling mode in get_next_action")

        if return_idx:
            return action.view(self.horizon, self.d_action), idx
        else:
            return action.view(self.horizon, self.d_action)

    def _save_data(self, idx, path):
        # Create the parent directory if it does not already exist.
        if not os.path.exists(path):
            os.makedirs(path)

        file_path = os.path.join(path, "{:03d}.npy".format(idx))
        particles = self.action_particles()
        np.save(file_path, particles.detach().cpu().numpy())

    def solve(self, state, shift_steps=1, n_iters=1, return_idx=False):
        """
        Optimize for best action at current state

        Parameters
        ----------
        state : torch.Tensor
            state to calculate optimal action from

        Returns
        -------
        action : torch.Tensor
            next action to execute
        """
        # get input device:
        inp_device = state.device
        inp_dtype = state.dtype
        state.to(**self.tensor_kwargs)
        self.state = state

        # Reset the optimizer before optimizing.
        self.init_optimizer()

        # Shift distribution to hotstart from previous timestep
        if self.hotstart and self.num_steps > 0:
            self._shift(shift_steps)
        else:
            self.reset_distribution()

        for i in range(n_iters):
            # Update the particles.
            self.optim.zero_grad()
            self.svgd.update()
            self.optim.step()

            # Reset so that the position and velocity reflect the accelerations exactly.
            u = self.svgd.particles().view(self.num_particles, self.horizon, self.d_action)[:, :, -self.dim:]
            particles = self.robot_model.rollout(u.contiguous(), self.state[..., None, :])
            self.svgd.reset(particles.view(self.num_particles, self.horizon * self.d_action))

        # Calculate best action
        curr_action_seq = self._get_action_seq(mode=self.sample_mode, return_idx=return_idx)

        self.num_steps += 1

        if return_idx:
            curr_action_seq, action_idx = curr_action_seq
            return curr_action_seq.to(inp_device, dtype=inp_dtype), action_idx
        else:
            return curr_action_seq.to(inp_device, dtype=inp_dtype)

    def _shift(self, shift_steps=1, repeat=False):
        """Predict distribution for the next time step by shifting the current
        mean forward by shift_steps."""
        p_size = (self.num_particles, self.horizon, self.dim)
        if self.shift_mode == "reset":
            u_particles = torch.normal(0, self.init_cov, p_size, **self.tensor_kwargs)
            shift_steps = 0  # Ensure not shifted.
        elif self.shift_mode == "best":
            # Get the chosen trajectory from the current particle set.
            seq, idx = self._get_action_seq(mode=self.sample_mode, return_idx=True)
            u_particles = seq.tile(self.num_particles, 1, 1)  # Repeat the selected trajectory for each particle.

            # Add some Gaussian noise.
            noise = torch.normal(0, self.init_cov, p_size, **self.tensor_kwargs)
            u_particles = u_particles + noise
        elif self.shift_mode == "resample":
            # Importance resample.
            num_draw = self.num_particles - 1
            u_particles = self.action_particles().view(*p_size)
            weights = self.svgd.calc_weights()**0.5
            indices = torch.multinomial(weights, num_draw, replacement=True)
            new_particles = u_particles[indices]
            noise = torch.normal(0, 0.01, (num_draw, self.horizon, self.dim), **self.tensor_kwargs)
            new_particles = new_particles + noise
            # Keep the best particle.
            u_particles = torch.concat((new_particles, u_particles[weights.argmax(), :, :].unsqueeze(0)))
        elif self.shift_mode == "shift":
            u_particles = self.action_particles().view(*p_size)
        else:
            raise Exception("SteinMPC: Unrecognized shift mode: " + self.shift_mode)

        # Shift the trajectories by the shift step.
        if shift_steps > 0:
            u_particles = u_particles.roll(-shift_steps, 1)
            # Fill the last elements of the trajectory.
            if repeat:
                u_particles[:, -shift_steps:, :] = u_particles[:, -shift_steps - 1, :].unsqueeze(1)
            else:
                # new_term = torch.normal(0, self.init_cov, (shift_steps, self.d_action), **self.tensor_kwargs)
                new_term = torch.zeros(1, **self.tensor_kwargs)
                u_particles[:, -shift_steps:, :] = new_term

        # Rollout the new particles and compute the states.
        particles = self.robot_model.rollout(u_particles.contiguous(), self.state[..., None, :])

        # Reset the distribution.
        self.svgd.reset(particles.view(self.num_particles, self.horizon * self.d_action))

    def reset_distribution(self, init_particles=None):
        """Reset control distribution."""
        if init_particles is None:
            p_size = (self.num_particles, self.horizon, self.dim)
            init_u = torch.normal(0, self.init_cov, p_size, **self.tensor_kwargs)
            init_particles = self.robot_model.rollout(init_u.contiguous(), self.state[..., None, :])
            init_particles = init_particles.view(self.num_particles, self.horizon * self.d_action)

        self.svgd.reset(init_particles)
        self.num_steps = 0
