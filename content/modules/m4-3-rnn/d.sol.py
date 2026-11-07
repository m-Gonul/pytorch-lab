import torch
from torch import nn


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        k = hidden_size ** -0.5
        self.weight_ih_l0 = nn.Parameter(torch.empty(hidden_size, input_size).uniform_(-k, k))
        self.weight_hh_l0 = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(-k, k))
        self.bias_ih_l0 = nn.Parameter(torch.empty(hidden_size).uniform_(-k, k))
        self.bias_hh_l0 = nn.Parameter(torch.empty(hidden_size).uniform_(-k, k))

    def forward(self, x, h0=None):
        N, T, _ = x.shape
        h = x.new_zeros(N, self.weight_hh_l0.shape[0]) if h0 is None else h0[0]
        xs = x @ self.weight_ih_l0.T + self.bias_ih_l0         # girdi kısmı tüm adımlar için tek matmul
        outs = []
        for t in range(T):
            h = torch.tanh(xs[:, t] + h @ self.weight_hh_l0.T + self.bias_hh_l0)
            outs.append(h)
        return torch.stack(outs, 1), h[None]


def input_grad_norms(rnn, x):
    """→ [‖∂(Σ h_T)/∂x_t‖ for t]"""
    x = x.detach().clone().requires_grad_(True)
    _, hn = rnn(x)
    g, = torch.autograd.grad(hn.sum(), x)
    return [g[:, t].norm().item() for t in range(x.shape[1])]
