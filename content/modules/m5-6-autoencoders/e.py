import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAE(nn.Module):
    def __init__(self, code=16):
        super().__init__()
        ...

    def forward(self, x):
        ...


class DenseAE(nn.Module):
    def __init__(self, code=16):
        super().__init__()
        ...

    def forward(self, x):
        ...
