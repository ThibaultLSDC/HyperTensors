import torch
from torch import arctanh
from hypertensors.gyrovectors import HyperTensor, l2norm


def dist(x: HyperTensor, y: HyperTensor) -> float:
    norm = l2norm((y - x).value)
    return float(2 * x.c**.5 * arctanh(x.c**.5 * norm))

def lam(x: HyperTensor, c: float=1.) -> HyperTensor:
    return 2 / (1 - c * l2norm(x)**2)