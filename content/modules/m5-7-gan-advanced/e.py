import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingNet(nn.Module):
    def __init__(self, z=64, w=64, L=8):
        super().__init__()
        ...

    def forward(self, z):
        ...


class AdaIN(nn.Module):
    def __init__(self, C, w):
        super().__init__()
        ...

    def forward(self, x, w):
        ...


class NoiseInjection(nn.Module):
    def __init__(self, C):
        super().__init__()
        ...

    def forward(self, x, noise=None):
        ...


def modulated_conv(x, weight, style, demod=True):
    ...


def truncate(w, w_avg, psi):
    ...
