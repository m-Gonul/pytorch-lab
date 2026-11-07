import torch


def sr_bf16(x, generator=None):
    """fp32 → bf16'ya stokastik yuvarlanmış fp32. Döngü yok."""
    i = x.contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, x.shape, generator=generator, dtype=torch.int32)
    return ((i + r) & -65536).view(torch.float32)


def train_bf16(X, y, lr, steps, rounding, generator=None):
    w = torch.ones(X.shape[1])
    rnd = (lambda t: t.to(torch.bfloat16).float()) if rounding == "nearest" else (lambda t: sr_bf16(t, generator))
    w = rnd(w)
    Xb = X.to(torch.bfloat16).float()
    for _ in range(steps):
        g = 2 * Xb.T @ (Xb @ w - y) / X.shape[0]
        w = rnd(w - lr * g)
    return ((X @ w - y) ** 2).mean().item()


g = torch.Generator().manual_seed(0)
v = torch.full((100000,), 1.001)
print(sr_bf16(v, g).mean(), v.to(torch.bfloat16).float()[0])   # ≈ 1.001 vs 1.0
