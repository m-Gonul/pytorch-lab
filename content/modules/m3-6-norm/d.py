import torch
from torch import nn


class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        ...

    def forward(self, x):
        ...


class MyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        ...

    def forward(self, x):
        ...
