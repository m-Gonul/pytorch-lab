import numpy as np


def _run(w, b, T, h0):
    hs = [h0]
    for _ in range(T):
        hs.append(w * np.tanh(hs[-1]) + b)
    return hs


def loss_grad(w, b, T, h0, target):
    hs = _run(w, b, T, h0)
    L = (hs[-1] - target) ** 2
    dh = 2 * (hs[-1] - target)
    dw = 0.0
    for t in range(T, 0, -1):
        dw += dh * np.tanh(hs[t - 1])
        dh = dh * w * (1 - np.tanh(hs[t - 1]) ** 2)
    return float(L), float(dw)


def find_wall(b, T, h0, lo, hi, iters=60):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _run(mid, b, T, h0)[-1] < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def step(w, g, lr, clip=None):
    if clip is not None:
        g = max(-clip, min(clip, g))
    return w - lr * g
