import torch
from torch import nn


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        ...

    def forward(self, x, h0=None):
        ...


def input_grad_norms(rnn, x):
    """→ [‖∂(Σ h_T)/∂x_t‖ for t]"""
    ...
