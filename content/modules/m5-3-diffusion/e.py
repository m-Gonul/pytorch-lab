import math
import torch
from torch import nn


def fm_loss(model, x1, z, t):
    """t: (B,) ∈ [0, 1]. Döngü yok."""
    ...


@torch.no_grad()
def euler_sample(model, z, steps):
    """Tek döngü: adımlar."""
    ...


mu, s = 3.0, 0.5
oracle = lambda x, t: mu + (t[:, None] * s ** 2 - (1 - t[:, None])) / ((1 - t[:, None]) ** 2 + t[:, None] ** 2 * s ** 2) * (x - t[:, None] * mu)
z = torch.randn(10000, 1)
for n in (1, 5, 50):
    x = euler_sample(oracle, z, n)
    print(n, x.mean().item(), x.std().item())
