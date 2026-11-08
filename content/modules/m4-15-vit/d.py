import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def digits16():
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32).view(-1, 1, 8, 8) / 16
    return F.interpolate(X, size=16, mode="bilinear", align_corners=False), torch.tensor(y)


class PatchEmbed(nn.Module):
    def __init__(self, P, C, D):
        super().__init__()
        ...

    def forward(self, x):
        ...

    def forward_unfold(self, x):
        ...


class ViT(nn.Module):
    def __init__(self, img=16, P=4, D=64, L=4, h=4, classes=10, pool="cls", pos=True):
        super().__init__()
        ...

    def forward(self, x):
        ...


def train_eval(model, Xtr, ytr, Xte, yte, epochs):
    ...
