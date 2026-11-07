import math
import torch
import torch.nn.functional as F
from torch import nn


def attention(q, k, v, causal=False):
    """(…, T, dk). F.scaled_dot_product_attention kullanmadan."""
    ...


class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        ...

    def forward(self, x, causal=False):
        ...


torch.manual_seed(0)
m = MultiHeadAttention(16, 4)
print(m(torch.randn(2, 7, 16), causal=True).shape)   # (2, 7, 16)
