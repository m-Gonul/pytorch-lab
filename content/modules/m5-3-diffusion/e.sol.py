import math
import torch
from torch import nn


def fm_loss(model, x1, z, t):
    """t: (B,) ∈ [0, 1]. Döngü yok."""
    tt = t[:, None]
    xt = (1 - tt) * z + tt * x1
    return ((model(xt, t) - (x1 - z)) ** 2).mean()


@torch.no_grad()
def euler_sample(model, z, steps):
    """Tek döngü: adımlar."""
    x = z.clone()
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, dtype=x.dtype)
        x = x + model(x, t) / steps
    return x


mu, s = 3.0, 0.5
oracle = lambda x, t: mu + (t[:, None] * s ** 2 - (1 - t[:, None])) / ((1 - t[:, None]) ** 2 + t[:, None] ** 2 * s ** 2) * (x - t[:, None] * mu)
z = torch.randn(10000, 1)
for n in (1, 5, 50):
    x = euler_sample(oracle, z, n)
    print(n, x.mean().item(), x.std().item())
