import torch


def sr_bf16(x, generator=None):
    """fp32 → bf16'ya stokastik yuvarlanmış fp32. Döngü yok."""
    ...


def train_bf16(X, y, lr, steps, rounding, generator=None):
    ...


g = torch.Generator().manual_seed(0)
v = torch.full((100000,), 1.001)
print(sr_bf16(v, g).mean(), v.to(torch.bfloat16).float()[0])   # ≈ 1.001 vs 1.0
