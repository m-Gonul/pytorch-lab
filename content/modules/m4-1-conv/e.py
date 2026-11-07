import torch
import torch.nn.functional as F


def conv2d_strided(x, w, b, stride=1, pad=0):
    """F.pad + iki .unfold (view) + einsum. Döngü yok."""
    ...
