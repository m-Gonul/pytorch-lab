import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_time_series(n, steps, seed):
    g = torch.Generator().manual_seed(seed)
    f1, f2, o1, o2 = torch.rand(4, n, 1, generator=g)
    t = torch.linspace(0, 1, steps)
    s = 0.5 * torch.sin((t - o1) * (f1 * 10 + 10)) + 0.2 * torch.sin((t - o2) * (f2 * 20 + 20))
    s = s + 0.1 * (torch.rand(n, steps, generator=g) - 0.5)
    return s[..., None]


def ex6_models():
    ...


def seq_targets(S, n_steps=40, ahead=8):
    ...


def train(m, X, Y, epochs=10, lr=1e-3):
    ...


def iterative_forecast(m, X, h, window=40):
    ...
