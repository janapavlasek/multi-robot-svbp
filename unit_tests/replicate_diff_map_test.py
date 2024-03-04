"""
Run tests to ensure that DiffMap implementation had been replicated
"""

import yaml
import numpy as np
import torch

from multi_robot_svbp.costs.obstacle_costs import SignedDistanceMap2DCost

"""
Old implementations
"""

HIGH = 1e6

class Rectangle_old(object):
    def __init__(self, w, h, c, tensor_kwargs={"device": "cpu", "dtype": torch.float32}):
        self.w = w
        self.h = h
        self.r = torch.tensor([w / 2, h / 2], **tensor_kwargs)
        self.c = torch.tensor(c, **tensor_kwargs)
        self.tensor_kwargs = tensor_kwargs

    def dist(self, p):
        p_rel = p - self.c
        q = (torch.abs(p_rel) - self.r)
        dist_out = torch.sum(q.clamp(min=0)**2, dim=-1)
        dist_in = q.max(dim=-1)[0].clamp(max=0)**2
        return dist_in - dist_out


class Circle_old(object):
    def __init__(self, r, c, tensor_kwargs={"device": "cpu", "dtype": torch.float32}):
        self.r = r
        self.c = torch.tensor(c, **tensor_kwargs)
        self.tensor_kwargs = tensor_kwargs

    def dist(self, p):
        p_rel = p - self.c
        return self.r**2 - torch.sum(p_rel**2, dim=-1)


class DiffMap_old(object):
    def __init__(self, file=None, width=None, height=None, origin=[0, 0], shapes=None,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}):
        self.width = width
        self.height = height
        self.origin = origin
        self.shapes = shapes if shapes is not None else []
        self.tensor_kwargs = tensor_kwargs

        if file is not None:
            self.load_from_file(file)

        assert self.width is not None and self.height is not None, "Must provide either width and height or map file."

        self.lims = [self.origin[0], self.origin[0] + self.width,
                     self.origin[1], self.origin[1] + self.height]

    def load_from_file(self, file_path):
        self.shapes = []

        with open(file_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.Loader)
            data = data["data"]
            self.width = data["width"]
            self.height = data["height"]
            self.origin = data["origin"]

            for key, val in data["obstacles"].items():
                if val["geometry"] == "rectangle":
                    self.add_rectangle(val["width"], val["height"], val["origin"])
                elif val["geometry"] == "circle":
                    self.add_circle(val["radius"], val["origin"])
                else:
                    print("WARNING: Unknown obstacle type:", val["geometry"])

    def add_circle(self, radius, center):
        self.shapes.append(Circle_old(radius, center, tensor_kwargs=self.tensor_kwargs))

    def add_rectangle(self, width, height, center):
        self.shapes.append(Rectangle_old(width, height, center, tensor_kwargs=self.tensor_kwargs))

    def compute_binary_img(self, ppm=40):
        sdf = self.compute_discrete_sdf(ppm=ppm)
        return sdf >= 0

    def compute_discrete_sdf(self, ppm=40):
        pix_w, pix_h, pts = self.grid_pts(ppm=ppm)

        sdf = torch.full((pix_w, pix_h), -HIGH, **self.tensor_kwargs)
        for shape in self.shapes:
            shape_sdf = shape.dist(pts).reshape((pix_w, pix_h))
            sdf = torch.maximum(shape_sdf, sdf)
        return sdf

    def grid_pts(self, ppm=40):
        pix_w, pix_h = int(ppm * self.width), int(ppm * self.height)
        x = np.linspace(*self.lims[:2], pix_w) + 1. / (2 * ppm)
        y = np.linspace(*self.lims[2:], pix_h) + 1. / (2 * ppm)
        X, Y = np.meshgrid(x, y)
        pts = np.stack([X.reshape(-1), np.flip(Y.reshape(-1))], axis=-1)
        pts = torch.tensor(pts, **self.tensor_kwargs)

        return pix_w, pix_h, pts

    def eval_sdf(self, x):
        dists = torch.stack([shape.dist(x) for shape in self.shapes])
        # TODO: Where this is positive, should be min, not max.
        return dists.max(dim=0)[0]

"""
Tests
"""

def test_diff_map_implm_replicated():
    batch_size = 6
    T = 3
    aabb_width = 0.2
    aabb_height = 0.4
    aabb_center = torch.randn(2)
    radius = 0.32
    circle_center = torch.randn(2)

    x = torch.randn(batch_size, T, 2, requires_grad=True)

    old_cost_fn = DiffMap_old(width=10.,height=10.,origin=[-5,-5])
    old_cost_fn.add_rectangle(aabb_width, aabb_height, aabb_center)
    # old_cost_fn.add_circle(radius,circle_center) -> old implementation is wrong!
    old_cost = old_cost_fn.eval_sdf(x)
    old_grad, = torch.autograd.grad(old_cost.sum(), x)

    new_cost_fn = SignedDistanceMap2DCost(())
    new_cost_fn.add_aabb(aabb_width, aabb_height, aabb_center)
    # new_cost_fn.add_circle(radius,circle_center)
    new_grad, new_cost = new_cost_fn.grad_w_cost(x)


    assert torch.allclose(old_cost, new_cost), "SignedDistMap2DCost cost output not equivalent to DiffMap output!"
    assert torch.allclose(old_grad, new_grad), "SignedDistMap2DCost grad output not equivalent to DiffMap output!"