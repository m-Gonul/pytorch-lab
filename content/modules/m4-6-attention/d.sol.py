import math
import torch
import torch.nn.functional as F
from torch import nn


def attention(q, k, v, causal=False):
    """(…, T, dk). F.scaled_dot_product_attention kullanmadan."""
    S = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        T, Tk = S.shape[-2:]
        S = S.masked_fill(torch.ones(T, Tk, dtype=torch.bool).triu(1), float("-inf"))
    return S.softmax(-1) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, causal=False):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, T, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        o = attention(q, k, v, causal)
        return self.proj(o.transpose(1, 2).reshape(B, T, d))


torch.manual_seed(0)
m = MultiHeadAttention(16, 4)
print(m(torch.randn(2, 7, 16), causal=True).shape)   # (2, 7, 16)
