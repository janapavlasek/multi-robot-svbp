import yaml
import numpy as np
import torch

from multi_robot_svbp.costs.obstacle_costs import SignedDistanceMap2DCost
HIGH = 1e6

class DiffMap(object):
    def __init__(self, file=None, width=None, height=None, origin=[0, 0], shapes=None,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}):
        self.width = width
        self.height = height
        self.origin = origin
        shapes = shapes if shapes is not None else []
        self.diff_map_fn = SignedDistanceMap2DCost(shapes, tensor_kwargs)
        self.tensor_kwargs = tensor_kwargs

        if file is not None:
            self.load_from_file(file)

        assert self.width is not None and self.height is not None, "Must provide either width and height or map file."

        self.lims = [self.origin[0], self.origin[0] + self.width,
                     self.origin[1], self.origin[1] + self.height]

    def load_from_file(self, file_path):
        with open(file_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.Loader)
            data = data["data"]
            self.width = data["width"]
            self.height = data["height"]
            self.origin = data["origin"]

            for key, val in data["obstacles"].items():
                if val["geometry"] == "rectangle":
                    self.add_rectangle(val["width"], val["height"], torch.tensor(val["origin"], **self.tensor_kwargs))
                elif val["geometry"] == "circle":
                    self.add_circle(val["radius"], torch.tensor(val["origin"], **self.tensor_kwargs))
                else:
                    print("WARNING: Unknown obstacle type:", val["geometry"])

    def add_circle(self, radius, center):
        self.diff_map_fn.add_circle(radius, center)

    def add_rectangle(self, width, height, center):
        self.diff_map_fn.add_aabb(width, height, center)

    def compute_binary_img(self, ppm=40):
        sdf = self.compute_discrete_sdf(ppm=ppm)
        return sdf >= 0

    def compute_discrete_sdf(self, ppm=40):
        pix_w, pix_h, pts = self.grid_pts(ppm=ppm)

        sdf = torch.full((pix_w, pix_h), -HIGH, **self.tensor_kwargs)
        sdf = torch.maximum(self.diff_map_fn.cost(pts).view(pix_w, pix_h), sdf)
        # for shape in self.shapes:
        #     shape_sdf = shape.dist(pts).reshape((pix_w, pix_h))
        #     sdf = torch.maximum(shape_sdf, sdf)
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
        return self.diff_map_fn.cost(x)

    def eval_grad_w_sdf(self, x):
        return self.diff_map_fn.grad_w_cost(x)
