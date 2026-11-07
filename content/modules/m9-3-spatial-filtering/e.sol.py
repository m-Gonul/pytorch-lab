import torch
import torch.nn.functional as F


def _win(a, k, dim):
    r = k // 2
    n = a.shape[dim]
    pad = [0, 0] * a.dim()
    pad[2 * (a.dim() - 1 - dim)] = r + 1
    pad[2 * (a.dim() - 1 - dim) + 1] = r
    cs = F.pad(a, pad).cumsum(dim)
    return cs.narrow(dim, k, n) - cs.narrow(dim, 0, n)


def box_cumsum(x, k):
    """x (B, H, W); sıfır doldurmalı k × k ortalama. Döngü yok."""
    xd = x.double()
    return (_win(_win(xd, k, 2), k, 1) / (k * k)).to(x.dtype)


x = torch.rand(2, 64, 64)
print(box_cumsum(x, 5).shape)
