import torch
from torch_bp.util.distances import pairwise_euclidean_distance
from typing import Iterable, Union
from math import floor


class TrajectoryDistance(object):
    def __init__(self, dim, horizon):
        self.dim = dim
        self.horizon = horizon

    def __call__(self, x, y):
        N, M = x.size(0), y.size(0)
        x = x.view(N, self.horizon, -1)[:, :, :self.dim]
        y = y.view(M, self.horizon, -1)[:, :, :self.dim]

        dist = pairwise_euclidean_distance(x, y).mean(-1)

        return dist


class SlidingWindowCrossDistance(object):
    """
    Given 2 trajectories, and a sliding window horizon, find
    1. Distance squred between all points in the sliding window
    2. Sum up distances squared for each window
    """
    def __init__(self, dim, horizon, sliding_window_horizon: int,
                 stride: int,
                 dims: Union[int, Iterable[int]] = (0,1)) -> None:
        """
        - Inputs:
            - sliding_window_horizon: int, size of each horizon
            - stride: int, stride of sliding window horizon
            - dims: int | (int,...), dimensions we are interested in comparing the norms
        """
        self.dim = dim
        self.horizon = horizon
        self.H = sliding_window_horizon
        self.S = stride
        self.dims = dims

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Calculate the sum distance squared for each window
        - Inputs:
            - x1: (...,K1,T,x_dim) tensor, 1st trajectory
            - x2: (...,K2,T,x_dim) tensor, 2nd trajectory
        - Outputs:
            - y: (...,floor{(T-H)/S}+1) tensor, sliding window cross distance
        """
        x1 = x1.view(-1, self.horizon, self.dim * 3)[:, :, :self.dim]
        x2 = x2.view(-1, self.horizon, self.dim * 3)[:, :, :self.dim]
        T = x1.shape[-2]
        N = floor((T-self.H)/self.S) + 1
        x1 = torch.stack([x1[...,self.S*i:self.S*i+self.H,self.dims] for i in range(N)], dim=-3)
        x2 = torch.stack([x2[...,self.S*i:self.S*i+self.H,self.dims] for i in range(N)], dim=-3)
        dist = (x1[...,:,None,:,:,:,None] -x2[...,None,:,:,:,None,:]).square().sum(dim=(-1,-2,-3,-4))

        return dist


def euclidean_path_length(path):
    """Calculates the length along a path. Path is a tensor with dimension (..., T, D)."""
    diff = path[..., :-1, :] - path[..., 1:, :]
    dist = (diff ** 2).sum(-1)  # (..., T, D) -> (..., T)
    return torch.sqrt(dist).sum(-1)
