import torch
from torch_bp.util.distances import pairwise_euclidean_distance
from typing import Iterable, Union
from math import floor
from typing import Any, Tuple


class DistanceFn(object):
    def __init__(self) -> None:
        self._jacrev = torch.func.jacrev(self._aux_fwd, argnums=(0,1), has_aux=True)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.forward(x, y)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise "Not Implemented"

    def backward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_x, d_y, dist = self._jacrev(x, y)
        return dist, d_x, d_y

    def _aux_fwd(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, y)
        return out, out

class TrajectoryDistance(DistanceFn):
    def __init__(self, dim: int, horizon: int) -> None:
        super().__init__()
        self.dim = dim
        self.horizon = horizon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        N, M = x.size(0), y.size(0)
        x = x.view(N, self.horizon, -1)[:, :, :self.dim]
        y = y.view(M, self.horizon, -1)[:, :, :self.dim]

        # delta = x.unsqueeze(-3) - y.unsqueeze(-4)
        # dist = (delta.unsqueeze(-2) @ delta.unsqueeze(-1)).squeeze((-1,-2)).mean(-1)

        dist = pairwise_euclidean_distance(x, y).mean(-1)

        return dist

    def backward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, M = x.size(0), y.size(0)
        full_dim = int(x.size(-1)/self.horizon)
        x = x.view(N, self.horizon, -1)[:, :, :self.dim]
        y = y.view(M, self.horizon, -1)[:, :, :self.dim]

        delta = x.unsqueeze(-3) - y.unsqueeze(-4) #(N,M,horizon,dim)
        dist = (delta.unsqueeze(-2) @ delta.unsqueeze(-1)).squeeze((-1,-2)).mean(-1) #(N,M)
        d_x = torch.cat(
            (2*delta/self.horizon,torch.zeros(N,M,self.horizon,full_dim-self.dim, device=x.device)),
            dim=-1) #(N,M,horizon,full_dim)
        d_y = -d_x
        return dist, d_x.view(N,M,-1), d_y.view(N,M,-1)

class SlidingWindowCrossDistance(DistanceFn):
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
        super().__init__()
        self.dim = dim
        self.horizon = horizon
        self.H = sliding_window_horizon
        self.S = stride
        self.dims = dims

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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


if __name__ == '__main__':
    dim = 3
    full_dim = 6
    horizon = 4
    N = 5
    M = 6

    x = torch.randn(N, horizon*full_dim)
    y = torch.randn(M, horizon*full_dim)

    dist_fn = TrajectoryDistance(dim, horizon)
    auto_fn = torch.func.jacrev(dist_fn.forward, argnums=(0,1))

    fwd_dist = dist_fn(x, y)
    fwd_dx, fwd_dy = auto_fn(x, y)

    bck_dist, bck_dx, bck_dy = dist_fn.backward(x, y)

    assert torch.allclose(fwd_dist, bck_dist)
    assert torch.allclose(fwd_dx.sum(-2), bck_dx)
    assert torch.allclose(fwd_dy.sum(-2), bck_dy)

    from torch_bp.inference.kernels import RBFMedianKernel

    gamma = 1. / torch.tensor([2 * horizon * dim]).sqrt()
    kernel = RBFMedianKernel(gamma=gamma, distance_fn=dist_fn)
    auto_fn = torch.func.jacrev(kernel.forward, argnums=(0,1))

    fwd_dist = kernel(x, y)
    fwd_dx, fwd_dy = auto_fn(x, y)

    bck_dist, bck_dx, bck_dy = kernel.backward(x, y)

    assert torch.allclose(fwd_dist, bck_dist)
    assert torch.allclose(fwd_dx.sum(-2), bck_dx)
    assert torch.allclose(fwd_dy.sum(-2), bck_dy)
