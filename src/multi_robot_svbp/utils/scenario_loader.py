"""
Scenario loader helper functions
"""

from typing import Dict
import os
import yaml
import numpy as np
import torch

def load_default_scenario(num_robots=5, env_perturb=1e-4,
                          tensor_kwargs={'device':'cpu', 'dtype':torch.float32}) -> Dict:
    """
    Loads a default scenario
    - Defintion:
        - N: number of robots for this scenario
        - pos_dim: dimension of pose, 2 (XY)
        - vel_dim: dimension of vel, 2 (XY)
        - x_dim: state dimension, 4 (pos_dim + vel_dim)
    - Inputs:
        - num_robots: int, number of robots to create scenario for
        - env_perturb: float, small perturbation in environment to break singularities
        - tensor_kwargs: dict, keyword args for instantiating tensors
    - Returns dict with following keys:
        - map_file: str, path to yaml file defining the map
        - num_robots: int, number of robots, N
        - start_state: (N, x_dim) tensor, starting 2D poses
        - goals: (N, x_dim) tensor, goal 2D poses
    """

    output = {}

    default_map = "data/squares.yml"
    output['map'] = os.path.join(os.path.expanduser(default_map))

    assert num_robots <= 8, "Default scenario defined up to 8 robots only."
    output['num_robots'] = num_robots

    docking_pts = torch.tensor([[1, 1], [5, 1], [9, 1], [1, 5], [9, 5], [1, 9], [5, 9], [9, 9]],
                               **tensor_kwargs)
    docking_pts += env_perturb * torch.randn(*docking_pts.shape, device=docking_pts.device) # small perturb in the goals and environment

    start_idx = np.random.choice(np.arange(docking_pts.shape[0]), num_robots, replace=False)
    goal_idx = np.random.choice(np.arange(docking_pts.shape[0]), num_robots, replace=False)

    output['goals'] = docking_pts[goal_idx]
    starts = docking_pts[start_idx]
    # Add zero velocities to start.
    output['start_state'] = torch.cat((starts, torch.zeros(num_robots, 2, **tensor_kwargs)), dim=1)

    return output


def load_yaml_scenario(scenario_path: str, env_perturb=1e-4,
                       tensor_kwargs={'device':'cpu', 'dtype':torch.float32}) -> Dict:
    """
    Loads scenario from yaml file
    - Defintion:
        - N: number of robots for this scenario
        - pos_dim: dimension of pose, 2 (XY)
        - vel_dim: dimension of vel, 2 (XY)
        - x_dim: state dimension, 4 (pos_dim + vel_dim)
    - Inputs:
        - scenario_path: str, full path where yaml file is at
        - env_perturb: float, small perturbation in environment to break singularities
        - tensor_kwargs: dict, keyword args for instantiating tensors
    - Returns dict with following keys:
        - map_file: str, path to yaml file defining the map
        - num_robots: int, number of robots, N
        - start_state: (N, x_dim) tensor, starting 2D poses
        - goals: (N, x_dim) tensor, goal 2D poses
    """

    output = {}

    # Read yaml scenario file
    with open(scenario_path, 'r') as scenario_file:
        scenario_data = [data for data in yaml.safe_load_all(scenario_file)]
    scenario_data = scenario_data[0]

    obs_file_path = scenario_data['map']
    output['map'] = os.path.join(os.path.expanduser(obs_file_path))
    num_robots = int(scenario_data['num_agents'])
    output['num_robots'] = num_robots
    start_state = torch.tensor(scenario_data['starts'], **tensor_kwargs)
    if env_perturb > 0:
        start_state += env_perturb * torch.randn(*start_state.shape, **tensor_kwargs) # small perturb in the environment
    output['start_state'] = torch.cat((start_state, torch.zeros_like(start_state)), dim=-1)
    goals = torch.tensor(scenario_data['goals'], **tensor_kwargs)
    if env_perturb > 0:
        goals += env_perturb * torch.randn(*start_state.shape, **tensor_kwargs) # small perturb in goals
    output['goals'] = goals

    return output
