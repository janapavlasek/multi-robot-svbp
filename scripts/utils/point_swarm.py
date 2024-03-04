import torch

from multi_robot_svbp.costs.base_costs import DimensionSumCost
from multi_robot_svbp.costs.obstacle_costs import SignedDistanceMap2DCost, KBendingObstacleCost
from multi_robot_svbp.costs.trajectory_costs import RunningDeltaCost, TerminalDeltaCost, StateBoundsCost


def make_costs(c_pos=0., c_vel=0.25, c_u=0.2, c_term=6., c_obs=50, obs_avoid_dist=.3,
               ctrl_space='acc', dim=2, horizon=1, goal=None, pos_lims=None, max_vel=None, max_acc=None,
               c_pos_bounds=1, c_vel_bounds=1, c_acc_bounds=1,
               signed_dist_map_fn: SignedDistanceMap2DCost = None,
               tensor_kwargs={"device": "cpu", "dtype": torch.float}):
    """Utility function for creating the costs for the point swarm problem."""
    obs_cost_fn = DimensionSumCost(cost_fn=KBendingObstacleCost(signed_2d_map=signed_dist_map_fn,
                                                                radius=obs_avoid_dist, k_bend=1.3,
                                                                sigma_T=torch.linspace(300., 250., horizon),
                                                                sigma=c_obs,
                                                                tensor_kwargs=tensor_kwargs),
                                   num_inp_dim=2, tensor_kwargs=tensor_kwargs)
    if ctrl_space == 'vel':
        running_cost_Qs = (c_pos * torch.eye(dim), c_vel * torch.eye(dim))
        running_cost_x_bars = (goal, torch.zeros_like(goal))

        terminal_cost_Qs = (c_term * torch.eye(dim), 0 * torch.eye(dim))
        terminal_cost_x_bars = (goal, torch.zeros_like(goal))
        max_acc = None  # Ensure no bounds on acceleration in velocity control.

    elif ctrl_space == 'acc':
        running_cost_Qs = (c_pos * torch.eye(dim), c_vel * torch.eye(dim), c_u * torch.eye(dim))
        running_cost_x_bars = (goal, torch.zeros_like(goal), torch.zeros_like(goal))

        terminal_cost_Qs = (c_term * torch.eye(dim), 0 * torch.eye(dim), 0 * torch.eye(dim))
        terminal_cost_x_bars = (goal, torch.zeros_like(goal), torch.zeros_like(goal))
    else:
        raise Exception("Unrecognized ctrl space")

    running_cost_fn = RunningDeltaCost(Qs=running_cost_Qs, x_bars=running_cost_x_bars,
                                       tensor_kwargs=tensor_kwargs)

    terminal_cost_fn = TerminalDeltaCost(Qs=terminal_cost_Qs, x_bars=terminal_cost_x_bars,
                                         tensor_kwargs=tensor_kwargs)

    bounds_cost_fn = StateBoundsCost(dim, c_pos=c_pos_bounds, c_vel=c_vel_bounds, c_acc=c_acc_bounds,
                                     pos_lims=pos_lims, max_vel=max_vel, max_acc=max_acc,
                                     tensor_kwargs=tensor_kwargs)

    return (obs_cost_fn, running_cost_fn, terminal_cost_fn, bounds_cost_fn)
