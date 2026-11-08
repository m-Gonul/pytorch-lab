import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def group_norm(x, G, eps=1e-5):
    C, H, W = x.shape
    g = x.reshape(G, -1)
    g = (g - g.mean(1, keepdims=True)) / np.sqrt(g.var(1, keepdims=True) + eps)
    return g.reshape(C, H, W)


def conv3x3(x, W, b):
    win = sliding_window_view(np.pad(x, ((0, 0), (1, 1), (1, 1))), (3, 3), axis=(1, 2))
    return np.einsum("chwuv,ocuv->ohw", win, W) + b[:, None, None]


def _silu(z):
    return z / (1 + np.exp(-z))


def resblock(x, temb, P):
    h = conv3x3(_silu(group_norm(x, P["G"])), P["W1"], P["b1"]) + (P["Wt"] @ temb + P["bt"])[:, None, None]
    h = conv3x3(_silu(group_norm(h, P["G"])), P["W2"], P["b2"])
    return x + h


def attn2d(x, P):
    C, H, W = x.shape
    n = group_norm(x, P["G"]).reshape(C, -1)
    q, k, v = P["Wq"] @ n, P["Wk"] @ n, P["Wv"] @ n
    s = q.T @ k / np.sqrt(C)
    a = np.exp(s - s.max(1, keepdims=True))
    a /= a.sum(1, keepdims=True)
    return x + (P["Wo"] @ (v @ a.T)).reshape(C, H, W)
