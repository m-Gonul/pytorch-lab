import math
import torch
import torch.nn.functional as F


def gradient_field(x, nbins=9):
    """x (B, H, W) → (M, theta, hist (B, nbins)). Döngü yok."""
    ...


x = torch.rand(2, 32, 32)
M, th, h = gradient_field(x)
print(M.shape, th.min().item() >= 0, h.shape)
