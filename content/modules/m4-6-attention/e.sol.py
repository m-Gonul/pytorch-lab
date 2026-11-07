import math
import torch
import torch.nn.functional as F


def blocked_attention(q, k, v, block=64, causal=False):
    """(…, T, dk). Tek döngü: key/value blokları. T×T skor matrisi yok."""
    *lead, T, dk = q.shape
    S = k.shape[-2]
    scale = 1 / math.sqrt(dk)
    m = q.new_full((*lead, T, 1), float("-inf"))
    l = q.new_zeros((*lead, T, 1))
    acc = q.new_zeros((*lead, T, v.shape[-1]))
    rows = torch.arange(T)[:, None]
    for s in range(0, S, block):
        kb, vb = k[..., s:s + block, :], v[..., s:s + block, :]
        sc = q @ kb.transpose(-2, -1) * scale
        if causal:
            cols = torch.arange(s, s + kb.shape[-2])[None, :]
            sc = sc.masked_fill(cols > rows, float("-inf"))
        m_new = torch.maximum(m, sc.amax(-1, keepdim=True))
        safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
        p = torch.exp(sc - safe)
        corr = torch.exp(m - safe)
        l = l * corr + p.sum(-1, keepdim=True)
        acc = acc * corr + p @ vb
        m = m_new
    return acc / l


g = torch.Generator().manual_seed(0)
q, k, v = (torch.randn(1, 2, 200, 16, generator=g) for _ in range(3))
print((blocked_attention(q, k, v, 64, causal=True) - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max())
