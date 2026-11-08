import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

T = 200
BETAS = torch.linspace(1e-4, 0.05, T)
AB = torch.cumprod(1 - BETAS, 0)


def digits16():
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32).view(-1, 1, 8, 8) / 16
    return F.interpolate(X, size=16, mode="bilinear", align_corners=False) * 2 - 1, torch.tensor(y)


def temb(t, d):
    f = torch.exp(-math.log(10000) * torch.arange(d // 2) / (d // 2))
    a = t[:, None].float() * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class ResBlock(nn.Module):
    def __init__(self, ci, co, e):
        super().__init__()
        ...

    def forward(self, x, e):
        ...


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        ...

    def forward(self, x):
        ...


class UNet(nn.Module):
    def __init__(self, c=16, C=10):
        super().__init__()
        ...

    def forward(self, x, t, y):
        ...


def train(model, steps, seed, ema_decay=0.99, p_uncond=0.1):
    ...


@torch.no_grad()
def sample(net, labels, s, steps, seed):
    ...
