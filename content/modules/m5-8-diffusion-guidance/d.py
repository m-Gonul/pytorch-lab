import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ANG = torch.arange(8) * math.pi / 4
MU = 4 * torch.stack([torch.cos(ANG), torch.sin(ANG)], 1)
LAB = torch.arange(8) % 4
T = 200
BETAS = torch.linspace(1e-4, 0.1, T)
AB = torch.cumprod(1 - BETAS, 0)


def data(n, g):
    k = torch.randint(0, 8, (n,), generator=g)
    return MU[k] + 0.5 * torch.randn(n, 2, generator=g), LAB[k]


class EpsNet(nn.Module):
    def __init__(self, C=4, h=128):
        super().__init__()
        ...

    def forward(self, x, t, y):
        ...


def train(model, steps, p_uncond=0.15, seed=0):
    ...


def sample(model, n, y, s, seed):
    ...
