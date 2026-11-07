import time
import torch


def _best(f, r=3):
    b = 1e9
    for _ in range(r):
        t0 = time.perf_counter()
        f()
        b = min(b, time.perf_counter() - t0)
    return b


def strided_cost(n, strides, repeat=3):
    x = torch.ones(n, dtype=torch.float32)
    out = []
    for s in strides:
        v = x[::s]
        t = _best(lambda: v.sum(), repeat)
        out.append(t / v.numel() * 1e9)
    return out


def transpose_variants(N):
    x = torch.randn(N, N)
    T = 32

    def tiled():
        return x.reshape(N // T, T, N // T, T).permute(2, 3, 0, 1).reshape(N, N)

    def rows():
        out = torch.empty(N, N)
        for i in range(N):
            out[:, i] = x[i]
        return out

    assert torch.equal(tiled(), x.t()) and torch.equal(rows(), x.t())
    return {"x.t().contiguous()": _best(lambda: x.t().contiguous()), "karo karo (T = 32)": _best(tiled), "Python satır döngüsü": _best(rows, 1)}
