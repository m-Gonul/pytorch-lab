import torch


def ema_scan(frames, alpha, L=64):
    T = frames.shape[0]
    a, b = 1 - alpha, alpha
    out = torch.empty_like(frames)
    carry = frames[0]
    out[0] = carry
    for s in range(1, T, L):
        X = frames[s:s + L]
        n = len(X)
        p = a ** torch.arange(1, n + 1, dtype=frames.dtype).view(-1, *[1] * (X.dim() - 1))
        local = torch.cumsum(b * X / p, 0)
        out[s:s + n] = p * (carry + local)
        carry = out[s + n - 1]
    return out


def ema_var_scan(frames, alpha, L=64, var0=100.0):
    B = ema_scan(frames, alpha, L)
    e = torch.empty_like(frames)
    e[0] = var0
    e[1:] = (frames[1:] - B[:-1]) ** 2
    return B, ema_scan(e, alpha, L)


def masks(frames, alpha, k=3.0):
    B, V = ema_var_scan(frames, alpha)
    m = torch.zeros_like(frames, dtype=torch.bool)
    m[1:] = (frames[1:] - B[:-1]).abs() > k * V[:-1].sqrt()
    return m


x = torch.rand(10, 4, 4, dtype=torch.float64)
print(ema_scan(x, 0.1, L=4).shape)
