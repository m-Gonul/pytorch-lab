import numpy as np


def rope(x, pos, base=10000):
    T, d = x.shape
    th = np.asarray(pos, float)[:, None] * base ** (-np.arange(0, d, 2) / d)[None]
    c, s = np.cos(th), np.sin(th)
    out = np.empty_like(x, dtype=float)
    out[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
    out[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
    return out
