import torch
import torch.nn.functional as F


def norm_over(x, dims, eps=1e-5):
    ...


def group_norm(x, G, gamma, beta, eps=1e-5):
    """x: (N, C, H, W). Döngü yok."""
    ...


x = torch.randn(2, 6, 4, 4)
print((group_norm(x, 3, torch.ones(6), torch.zeros(6)) - F.group_norm(x, 3)).abs().max())
