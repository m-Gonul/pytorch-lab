import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        ...

    def forward(self, x):
        ...


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        ...

    def forward(self, x):
        ...


class Inception(nn.Module):
    def __init__(self, cin, c1, c3r, c3, c5r, c5, pp):
        super().__init__()
        ...

    def forward(self, x):
        ...


def ex5_models():
    ...


def shifted_digits(n_rep, seed):
    ...


def train_eval(model, X, y, Xt, yt, epochs=8):
    ...
