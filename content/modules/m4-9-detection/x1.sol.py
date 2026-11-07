import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def dense_to_conv(W, C, k):
    return W.reshape(W.shape[0], C, k, k)


def fcn_apply(F, Wc, b):
    k = Wc.shape[-1]
    win = sliding_window_view(F, (k, k), axis=(1, 2))
    return np.einsum("chwuv,ocuv->ohw", win, Wc) + b[:, None, None]


def sliding_dense(F, W, b, k):
    C, H, Wd = F.shape
    win = sliding_window_view(F, (k, k), axis=(1, 2))
    flat = win.transpose(1, 2, 0, 3, 4).reshape(H - k + 1, Wd - k + 1, C * k * k)
    return (flat @ W.T + b).transpose(2, 0, 1)
