import math
import torch
import torch.nn as nn


class BatchedMLP(nn.Module):
    def __init__(self, E, d_in, h, depth, d_out):
        super().__init__()
        ...

    def forward(self, X):
        ...


def train_batched(E, fn, depth, steps=1500, n=1000):
    ...


def train_loop(E, fn, depth, steps=1500, n=1000):
    ...
