import torch


def rec_loop(a, b):
    x = torch.empty_like(b)
    acc = torch.zeros_like(b[:, 0])
    for t in range(b.shape[1]):
        acc = a[:, t] * acc + b[:, t]
        x[:, t] = acc
    return x


def rec_scan(a, b):
    A, B = a.clone(), b.clone()
    s = 1
    while s < A.shape[1]:
        A2 = A[:, s:] * A[:, :-s]
        B2 = A[:, s:] * B[:, :-s] + B[:, s:]
        A = torch.cat([A[:, :s], A2], 1)
        B = torch.cat([B[:, :s], B2], 1)
        s *= 2
    return B


def rec_cumprod(a, b):
    P = torch.cumprod(a, 1)
    return P * torch.cumsum(b / P, 1)


def compact_scan(x, keep):
    f = keep.long()
    pos = torch.cumsum(f, 0) - f
    n = int(f.sum())
    out = torch.empty(n + 1, dtype=x.dtype)
    dest = torch.where(keep, pos, torch.full_like(pos, n))
    out.scatter_(0, dest, x)
    return out[:n]
