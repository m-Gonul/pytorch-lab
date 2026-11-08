import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def digits16():
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32).view(-1, 1, 8, 8) / 16
    return F.interpolate(X, size=16, mode="bilinear", align_corners=False) * 2 - 1, torch.tensor(y)


class Generator(nn.Module):
    def __init__(self, z=32, c=64):
        super().__init__()
        ...

    def forward(self, z):
        ...


class Discriminator(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        ...

    def forward(self, x):
        ...


def train_gan(G, D, X, epochs, z=32, lr=2e-4):
    ...


def fid_np(F1, F2):
    ...
