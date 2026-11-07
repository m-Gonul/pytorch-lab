import torch


def quantize_int4(W, group):
    """→ (q int8 [-8, 7], s (out, in/group)). Döngü yok."""
    out, inp = W.shape
    G = W.reshape(out, inp // group, group)
    s = G.abs().amax(-1).clamp_min(1e-12) / 7
    q = torch.clamp(torch.round(G / s[..., None]), -8, 7).to(torch.int8)
    return q.reshape(out, inp), s


def pack_int4(q):
    lo = q[..., 0::2].to(torch.int16) & 0xF
    hi = q[..., 1::2].to(torch.int16) & 0xF
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_int4(p):
    v = p.to(torch.int16)
    lo, hi = v & 0xF, (v >> 4) & 0xF
    both = torch.stack([lo, hi], -1).reshape(*p.shape[:-1], -1)
    return torch.where(both > 7, both - 16, both).to(torch.int8)


def dequant_int4(p, s, group):
    q = unpack_int4(p).float()
    out, inp = q.shape
    return (q.reshape(out, inp // group, group) * s[..., None]).reshape(out, inp)


torch.manual_seed(0)
W = torch.randn(64, 256)
q, s = quantize_int4(W, 32)
p = pack_int4(q)
print(p.dtype, p.shape, (dequant_int4(p, s, 32) - W).abs().max())
