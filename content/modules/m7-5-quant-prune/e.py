import torch


def quantize_int4(W, group):
    """→ (q int8 [-8, 7], s (out, in/group)). Döngü yok."""
    ...


def pack_int4(q):
    ...


def unpack_int4(p):
    ...


def dequant_int4(p, s, group):
    ...


torch.manual_seed(0)
W = torch.randn(64, 256)
q, s = quantize_int4(W, 32)
p = pack_int4(q)
print(p.dtype, p.shape, (dequant_int4(p, s, 32) - W).abs().max())
