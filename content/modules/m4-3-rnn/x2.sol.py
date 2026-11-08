import numpy as np


def forward(w, u, x):
    h = [0.0]
    for xt in x:
        h.append(w * h[-1] + u * xt)
    return h


def grad_w(w, u, x, y, k=None):
    h = forward(w, u, x)
    T = len(x)
    lo = 1 if k is None else max(1, T - k + 1)
    return (h[-1] - y) * sum(w ** (T - t) * h[t - 1] for t in range(lo, T + 1))
