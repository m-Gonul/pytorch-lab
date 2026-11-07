import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, cin, cout, k, d=1):
        ...

    def forward(self, x):
        ...


class WaveNet(nn.Module):
    def __init__(self, c=16, dilations=(1, 2, 4, 8) * 2, out=8):
        super().__init__()
        ...

    def forward(self, x):
        ...
