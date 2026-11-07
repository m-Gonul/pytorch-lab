import torch
from torch import nn


class SoftStump(nn.Module):
    def __init__(self, d, C, tau=0.5):
        super().__init__()
        ...

    def forward(self, x):
        ...


def train_soft_stump(X, y, C, steps=500, lr=0.1):
    ...


torch.manual_seed(0)
X = torch.rand(300, 4)
y = (X[:, 2] > 0.6).long()
m = train_soft_stump(X, y, 2)
print((m(X).argmax(1) == y).float().mean())
