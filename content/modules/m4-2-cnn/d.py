import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        ...

    def forward(self, x):
        ...


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        ...

    def forward(self, x):
        ...
