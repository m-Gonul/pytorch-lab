import torch
from torch import nn


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        k = hidden_size ** -0.5
        u = lambda *s: nn.Parameter(torch.empty(*s).uniform_(-k, k))
        self.hidden_size = hidden_size
        self.weight_ih = u(4 * hidden_size, input_size)
        self.weight_hh = u(4 * hidden_size, hidden_size)
        self.bias_ih = u(4 * hidden_size)
        self.bias_hh = u(4 * hidden_size)

    def forward(self, x, state=None):
        if state is None:
            z = x.new_zeros(x.shape[0], self.hidden_size)
            state = (z, z)
        h, c = state
        a = x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
        i, f, g, o = a.chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


def run_cell(cell, x):
    """x: (N, T, D) → (out (N, T, H), (h_T, c_T))"""
    state, outs = None, []
    for t in range(x.shape[1]):
        state = cell(x[:, t], state)
        outs.append(state[0])
    return torch.stack(outs, 1), state
