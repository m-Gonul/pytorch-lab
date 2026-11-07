import math
import torch
import torch.nn.functional as F


def blocked_attention(q, k, v, block=64, causal=False):
    """(…, T, dk). Tek döngü: key/value blokları. T×T skor matrisi yok."""
    ...


g = torch.Generator().manual_seed(0)
q, k, v = (torch.randn(1, 2, 200, 16, generator=g) for _ in range(3))
print((blocked_attention(q, k, v, 64, causal=True) - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max())
