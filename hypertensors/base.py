from typing import Any
import torch
from torch import tanh, arctanh, sqrt
from functools import partial


l2norm = partial(torch.norm, p=2)


class HyperTensor(torch.Tensor):
    def __init__(self, tensor: torch.Tensor, curvature=1.):

        htensor = self.exponential_map(tensor, curvature)
        super().__init__(htensor)

    @classmethod
    def exponential_map(cls, t, c):
        return tanh(sqrt(c) * l2norm(t)) * t / (sqrt(c) * l2norm(t))

    @classmethod
    def logarithmic_map(cls, t, c):
        return arctanh(sqrt(c) * l2norm(t)) * t / (sqrt(c) * l2norm(t))

    def __add__(self, other: Any):
        return super().__add__(other)