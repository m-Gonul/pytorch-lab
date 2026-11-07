import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _win(x, K):
    r = K // 2
    return sliding_window_view(np.pad(x, ((0, 0), (r, r), (r, r))), (K, K), axis=(1, 2))


def conv2d(x, w):
    return np.einsum("chwuv,ocuv->ohw", _win(x, w.shape[-1]), w)


def depthwise(x, w):
    return np.einsum("chwuv,cuv->chw", _win(x, w.shape[-1]), w)


def pointwise(x, w):
    return np.einsum("oc,chw->ohw", w, x)


def grouped_conv(x, w, g):
    C, H, W = x.shape
    Cout, cg, K, _ = w.shape
    xw = _win(x, K).reshape(g, C // g, H, W, K, K)
    wg = w.reshape(g, Cout // g, cg, K, K)
    return np.einsum("gchwuv,gocuv->gohw", xw, wg).reshape(Cout, H, W)


def se_block(x, W1, b1, W2, b2):
    s = x.mean((1, 2))
    z = np.maximum(0, W1 @ s + b1)
    e = 1 / (1 + np.exp(-(W2 @ z + b2)))
    return x * e[:, None, None]


def lrn(x, k=2, n=5, alpha=1e-4, beta=0.75):
    C = x.shape[0]
    sq = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x ** 2, 0)])
    lo = np.clip(np.arange(C) - n // 2, 0, C)
    hi = np.clip(np.arange(C) + n // 2 + 1, 0, C)
    s = sq[hi] - sq[lo]
    return x / (k + alpha * s) ** beta
