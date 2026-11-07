import torch
import torch.nn.functional as F


def box_cumsum(x, k):
    """x (B, H, W); sıfır doldurmalı k × k ortalama. Döngü yok."""
    ...


x = torch.rand(2, 64, 64)
print(box_cumsum(x, 5).shape)
