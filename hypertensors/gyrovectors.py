from __future__ import annotations
from typing import Any, List, Tuple, Union

import torch

from torch import tanh, arctanh
from math import sqrt
from functools import partial

l2norm = partial(torch.norm, p=2, dim=-1)


class HyperTensor:
    def __init__(self, tensor: torch.Tensor, curvature: float=1.):    
        self.value = tensor
        self.c = curvature

    def from_real(self, tensor: torch.Tensor, curvature: float=1.):
        self.value = self.exponential_map(tensor, curvature)
        self.c = curvature

    @staticmethod
    def exponential_map(t, c):
        return tanh(sqrt(c) * l2norm(t)) * t / (sqrt(c) * l2norm(t) + 1e-9)

    @staticmethod
    def logarithmic_map(t, c):
        return arctanh(sqrt(c) * l2norm(t)) * t / (sqrt(c) * l2norm(t) + 1e-9)

    @staticmethod
    def zeros(shape: Union[Tuple, List], curvature=1.) -> HyperTensor:
        return HyperTensor(torch.zeros(shape), curvature=curvature)
    
    def __add__(self, other: HyperTensor) -> HyperTensor:
        """
        Möbius addition, only 1D vector at the moment
        """
        inner = self.value @ other.value
        normx = l2norm(self.value)
        normy = l2norm(other.value)

        a = (1 + 2 * self.c * inner + self.c * normy**2) * self.value
        b = (1 - self.c * normx**2) * other.value
        c = 1 + 2 * self.c * inner + self.c**2 * normx**2 * normy**2

        return HyperTensor((a + b) / c, curvature= self.c)

    def __sub__(self, other: HyperTensor) -> HyperTensor:
        return self + HyperTensor(-other.value, self.c)
    
    def __mul__(self, other: float) -> HyperTensor:
        """
        Möbius scalar (right) multiplication, should support any dim
        """
        tan = tanh(other * arctanh(sqrt(self.c) * l2norm(self.value)))
        res = tan * self.value / l2norm(self.value) / sqrt(self.c)
        return HyperTensor(res, self.c)

    def __rmul__(self, other: float) -> HyperTensor:
        """
        Möbius scalar (left) multiplication, should support any dim
        """
        return self * other
    
    def __neg__(self) -> HyperTensor:
        """
        Special cace of Möbius scalar mult
        """
        return -1. * self

    def __truediv__(self, other: float) -> HyperTensor:
        return self * (1 / other)

    def __repr__(self) -> str:
        return f"HyperTensor{self.value.__repr__()[6:]}"