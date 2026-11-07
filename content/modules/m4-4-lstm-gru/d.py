import torch
from torch import nn


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        ...

    def forward(self, x, state=None):
        ...


def run_cell(cell, x):
    """x: (N, T, D) → (out (N, T, H), (h_T, c_T))"""
    ...
