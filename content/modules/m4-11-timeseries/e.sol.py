import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, cin, cout, k, d=1):
        super().__init__(cin, cout, k, dilation=d)
        self.left = (k - 1) * d

    def forward(self, x):
        return super().forward(F.pad(x, (self.left, 0)))


class _Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.f, self.g = CausalConv1d(c, c, 2, d), CausalConv1d(c, c, 2, d)
        self.res, self.skip = nn.Conv1d(c, c, 1), nn.Conv1d(c, c, 1)

    def forward(self, x):
        z = torch.tanh(self.f(x)) * torch.sigmoid(self.g(x))
        return x + self.res(z), self.skip(z)


class WaveNet(nn.Module):
    def __init__(self, c=16, dilations=(1, 2, 4, 8) * 2, out=8):
        super().__init__()
        self.inp = CausalConv1d(1, c, 2)
        self.blocks = nn.ModuleList([_Block(c, d) for d in dilations])
        self.head = nn.Sequential(nn.ReLU(), nn.Conv1d(c, c, 1), nn.ReLU(), nn.Conv1d(c, out, 1))

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        skip = 0
        for b in self.blocks:
            h, s = b(h)
            skip = skip + s
        return self.head(skip).transpose(1, 2)
