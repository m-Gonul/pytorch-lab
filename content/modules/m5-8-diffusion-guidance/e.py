import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 200
BETAS = torch.linspace(1e-4, 0.1, T)
AB = torch.cumprod(1 - BETAS, 0)


class CondDenoiser(nn.Module):
    def __init__(self, D, C=10, h=256):
        super().__init__()
        ...

    def forward(self, x, t, y):
        ...


def train_diffusion(Z, y, steps, seed):
    ...


def sample(model, labels, D, s, seed):
    ...
