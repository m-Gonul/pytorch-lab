import copy
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch import nn


class QuantLinear(nn.Module):
    @classmethod
    def from_float(cls, linear):
        ...

    def forward(self, x):
        ...


def quantize_model(model):
    ...


def state_bytes(model):
    ...


def prune_global(model, amount):
    ...


torch.manual_seed(0)
m = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
q = quantize_model(m)
x = torch.randn(4, 128)
print((q(x) - m(x)).abs().max(), state_bytes(m), state_bytes(q))
