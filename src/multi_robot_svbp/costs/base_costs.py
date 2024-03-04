"""
Base costs module
- contains core cost functions used to build or compose other cost functions together to create
    more complex cost functions with gradient
"""

import torch
from typing import Iterable, Tuple, Union


class BaseCost(object):
    """
    Base class for all cost modules to unify their call sequence under a common interface
    """
    def __init__(self, tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Inputs:
            - tensor_kwargs : dict of tensor keyword args to be used when creating new tensors
        """
        self.tensor_kwargs = tensor_kwargs

    def __call__(self, *args, grad=False) -> Union[torch.Tensor, Tuple[torch.Tensor,...]]:
        """
        - Inputs:
            - *args : arguments to be passed to cost() or grad_w_cost() function call
        """
        return self.grad_w_cost(*args) if grad else self.cost(*args)

    def cost(self, *args) -> torch.Tensor:
        """
        - Inputs:
            - *args : arguments to be used to calculate cost
        - Returns:
            - cost : tensor
        """
        raise NotImplementedError("BaseCost subclass to implement cost method!")

    def grad_w_cost(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        The default version of any cost uses autograd to compute the gradients.
        It is more efficient to manually compute the gradients.
        - Inputs:
            - *args : arguments to be used to calculate cost
        - Returns:
            - grad(s) : tensors,...
            - cost : tensor
        """
        x = x.detach().requires_grad_(True)
        cost = self.cost(x)
        grad, = torch.autograd.grad(cost.sum(), [x])
        return grad, cost


class LinearCost(BaseCost):
    """
    A standard expression of cost in the form of c = Q x, where c is a scalar,
    Q : tensor (y_dim, x_dim)

    NOTE: cost is can be not scalar depending on dim of Q
    """
    def __init__(self, Q : torch.Tensor, sigma = 1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Inputs:
            - Q : tensor (x_dim, x_dim) square matrix
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        assert len(Q.shape) == 2, "Q must be 2 dimensional to not be ambiguous"
        self.Q = Q.to(**self.tensor_kwargs)
        self.sigma = sigma

    def cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - x : tensor (...,x_dim)
        - Returns:
            - cost : tensor (...,y_dim) or (...,) if y_dim = 1
        """
        return self.sigma * (self.Q @ x[...,:,None])[...,0].squeeze(dim=-1)

    def grad_w_cost(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Inputs:
            - x : tensor (...,x_dim)
        - Returns:
            - grad : tensor (...,y_dim,x_dim) or (...,x_dim) if y_dim =1
            - cost : tensor (...,y_dim) or (...,) if y_dim = 1
        """
        batch_shape = x.shape[:-1]
        cost =  self.cost(x)
        grad =  self.sigma * self.Q.expand(*batch_shape, *self.Q.shape).squeeze(dim=-2)
        return grad, cost


class QuadraticCost(BaseCost):
    """
    A standard expression of cost in the form of c = x^T Q x, where c is a scalar,
    Q : tensor (x_dim,x_dim)
    """
    def __init__(self, Q : torch.Tensor, sigma = 1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Inputs:
            - Q : tensor (x_dim, x_dim) square matrix
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        assert Q.shape[-1] == Q.shape[-2], f"Q needs to be square matrix, currently Q shape = [{Q.shape}]"
        self.Q = Q.to(**self.tensor_kwargs)
        self.sigma = sigma

    def cost(self, x : torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - x : tensor (...,x_dim)
        - Returns:
            - cost : tensor (...,)
        """
        return self.sigma * (x[...,None,:] @ self.Q @ x[...,:,None])[...,0,0]

    def grad_w_cost(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Inputs:
            - x : tensor (...,x_dim)
        - Returns:
            - grad : tensor (...,x_dim)
            - cost : tensor (...,)
        """
        cost = self.cost(x)
        grad = self.sigma * (x[...,None,:] @ (self.Q.T + self.Q))[...,0,:] # x^T (Q^T + Q)
        return grad, cost


class BoundCost(BaseCost):

    """A cost for keeping values inside bounds."""

    def __init__(self, bounds, sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        Inputs:
            - bounds: tensor (2, x_dim) where the first row is the lower bound
                      and the second row is the upper bound.
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__()
        self.bounds = torch.as_tensor(bounds, **tensor_kwargs)
        self.sigma = sigma

    def cost(self, x: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - x : tensor (...,x_dim)
        - Returns:
            - cost : tensor (...,)
        """
        lower = torch.square((x - self.bounds[0, :]).clamp(max=0))
        upper = torch.square((x - self.bounds[1, :]).clamp(min=0))
        cost = torch.maximum(lower, upper).sum(-1)
        return self.sigma * cost


class CompositeMaxCost(BaseCost):
    """
    A composite class that composes a few scalar cost functions and optionally a scalar together and take their max

    NOTE: for now only works on scalar costs
    """
    def __init__(self, costs : Iterable[BaseCost], opt_scalar=float('-inf'), sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Input:
            - costs : cost objects whose outputs would be maxed against each other
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        self.costs = [*costs]
        self.opt_scalar = opt_scalar
        self.sigma = sigma

    def cost(self, *args) -> torch.Tensor:
        """
        - Input:
            - *args : one or more tensors
        - Returns:
            - cost : tensor (...,), since output is scalar, only batch shape is retained
        """
        return self.sigma * torch.stack([cost.cost(*args) for cost in self.costs]).max(dim=0)[0].clamp(min=self.opt_scalar)

    def grad_w_cost(self, *args) -> Tuple[torch.Tensor,...]:
        """
        - Input:
            - *args : one or more tensors
        - Returns:
            - *grads : tensors, grads with same shape as input args
            - cost : tensor (...,), since output is scalar, only batch shape is retained
        """
        grads_w_costs = [torch.stack(t) for t in zip(*[cost.grad_w_cost(*args) for cost in self.costs])]
        gradss = grads_w_costs[:-1]
        costs = grads_w_costs[-1]
        # for i,grads_i in enumerate(gradss):
        #     gradss[i] = torch.cat((grads_i, torch.zeros_like(grads_i[0,...][None,...])))
        # costs = torch.cat((costs, self.opt_scalar * torch.ones_like(costs[0,...])[None,...]))
        cost, cost_ind = costs.max(dim=0, keepdim=True)
        t = [...] + [None for _ in range(gradss[0].dim() - cost.dim())]
        grads = [self.sigma * grads_i.take_along_dim(cost_ind[t], dim=0)[0,...] for grads_i in gradss]
        cost_less = cost < self.opt_scalar
        cost[cost_less] = self.opt_scalar
        for i,grads_i in enumerate(grads):
            grads[i] = grads_i * ~cost_less[0,...][t]
        return *grads, self.sigma * cost[0,...]


class CompositeSumCost(BaseCost):
    """
    A composite class that composes a few scalar cost functions and optionally a scalar together and take their sum

    NOTE: for now only works on scalar costs
    """
    def __init__(self, costs : Iterable[BaseCost], add_scalar=0.0, sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Input:
            - costs : cost objects whose outputs would be summed together
            - add_scalar : optional additional scalar to be added to the sum of costs
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        self.costs = [*costs]
        self.add_scalar = add_scalar
        self.sigma = sigma

    def cost(self, *args) -> torch.Tensor:
        """
        - Input:
            - *args : one or more tensors
        - Returns:
            - cost : tensor (...,), since output is scalar, only batch shape is retained
        """
        return self.sigma * (torch.stack([cost.cost(*args) for cost in self.costs]).sum(dim=0) + self.add_scalar)

    def grad_w_cost(self, *args) -> Tuple[torch.Tensor,...]:
        """
        - Input:
            - *args : one or more tensors
        - Returns:
            - *grads : tensors, grads with same shape as input args
            - cost : tensor (...,), since output is scalar, only batch shape is retained
        """
        grads_w_costs = [torch.stack(t) for t in zip(*[cost.grad_w_cost(*args) for cost in self.costs])]
        gradss = torch.stack(grads_w_costs[:-1])
        cost = self.sigma * (grads_w_costs[-1].sum(dim=0) + self.add_scalar)
        grads = self.sigma * gradss.sum(dim=1)
        return *grads, cost


class DimensionSumCost(BaseCost):
    """
    A class that takes in a cost object and sums the output along the specified dimensions
    - Trivia: This thing was THIS close to being called 'DimSum' cause why not
    """
    def __init__(self, cost_fn : BaseCost,
                 num_inp_dim: Union[int, Iterable[int]],
                 dim : Union[int, Iterable[int]]=-1, sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Input:
            - cost : cost objects whose outputs would be summed
            - num_inp_dim: int | Iterable[int], number of input dimension of each args
                -> f(x_1,x_2) such that x_1 = (....,dim_1,dim_2), x_2 = (...,dim_3) => num_inp_dim = (2,1)
            - dim : int | Iterable[int], dimensions or iterable of dimensions to be summed
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        dims = dim if isinstance(dim, Iterable) else (dim,)
        for dim_i in dims:
            assert dim_i < 0, f"For now only accept dim < 0, since we have no idea what the input batch size would be"
        self.cost_fn = cost_fn
        self.dims = dims
        num_inp_dims = num_inp_dim if isinstance(num_inp_dim, Iterable) else (num_inp_dim,)
        self.jac_dims = [[dim_i - num_inp_dim_i for dim_i in dims] for num_inp_dim_i in num_inp_dims]
        self.sigma = sigma

    def cost(self, *args) -> torch.Tensor:
        """
        - Input:
            - *args : one or more tensors (...,*in_dim_i) for i=1,...N where N is num of inputs
        - Returns:
            - cost : tensor (...,*out_dim)
        """
        return self.sigma * self.cost_fn.cost(*args).sum(self.dims)

    def grad_w_cost(self, *args) -> Tuple[torch.Tensor,...]:
        """
        - Input:
            - *args : one or more tensors (...,*in_dim_i) for i=1,...N where N is num of inputs
        - Returns:
            - *grads : tensors (...,*out_dim,*in_dim_i) for i=1,...N where N is num of input
            - cost : tensor (...,*out_dim)
        """
        grads_w_cost = self.cost_fn.grad_w_cost(*args)
        grads = [self.sigma * grad_i.sum(dim=jac_dim) for jac_dim, grad_i in zip(self.jac_dims, grads_w_cost)]
        cost = self.sigma * grads_w_cost[-1].sum(self.dims)
        return *grads, cost


class ExponentialCost(BaseCost):
    """
    A class that takes in a cost object and exponantiate it
    """
    def __init__(self, log_cost : BaseCost, sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        - Input:
            - log_cost : cost objects whose outputs would be exponantiated
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__(tensor_kwargs)
        self.log_cost = log_cost
        self.sigma = sigma

    def cost(self, *args) -> torch.Tensor:
        """
        - Input:
            - *args : one or more tensors
        - Returns:
            - cost : tensor (...,), since output is scalar, only batch shape is retained
        """
        return self.sigma * self.log_cost.cost(*args).exp()

    def grad_w_cost(self, *args) -> Tuple[torch.Tensor,...]:
        """
        - Input:
            - *args : one or more tensors
        - Returns:
            - *grads : tensors, grads with same shape as input args
            - cost : tensor (...,), since output is scalar, only batch shape is retained
        """
        log_grads_w_cost = self.log_cost.grad_w_cost(*args)
        log_grads = log_grads_w_cost[:-1]
        cost = log_grads_w_cost[-1].exp()
        t = [...] + [None for _ in range(log_grads[0].dim()-cost.dim())]
        return *[self.sigma * log_grad * cost[t] for log_grad in log_grads], self.sigma * cost


class PreEvaluatedCost(BaseCost):
    """
    Special cost function whose values can be evaluated beforehand and stored,
    Useful for cases where the same function would be repeatedly evaluated
    """
    def __init__(self, cost_fn : BaseCost,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        super().__init__(tensor_kwargs)
        self.cost_fn = cost_fn
        self.stored_cost = None
        self.stored_grads = None

    def cost(self, *args) -> torch.Tensor:
        return self.stored_cost

    def grad_w_cost(self, *args) -> Tuple:
        return *self.stored_grads, self.stored_cost

    def pre_eval_cost(self, *args) -> None:
        self.stored_cost = self.cost_fn.cost(*args)

    def pre_eval_grad_w_cost(self, *args) -> None:
        grad_w_cost = self.cost_fn.grad_w_cost(*args)
        self.stored_cost = grad_w_cost[-1]
        self.stored_grads = grad_w_cost[:-1]
