import numpy as np


def _windows(x, KH, KW, stride, pad):
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (KH, KW), axis=(2, 3))   # (N, C, H', W', KH, KW)
    return win[:, :, ::stride, ::stride]


def conv2d_np(x, w, b, stride=1, pad=0):
    """x: (N, C, H, W), w: (F, C, KH, KW) → (N, F, OH, OW). Döngü yok."""
    win = _windows(x, w.shape[2], w.shape[3], stride, pad)
    return np.einsum("nchwkl,fckl->nfhw", win, w) + b[None, :, None, None]


def im2col_np(x, KH, KW, stride=1, pad=0):
    """→ (N, C·KH·KW, OH·OW). Döngü yok."""
    win = _windows(x, KH, KW, stride, pad)                     # (N, C, OH, OW, KH, KW)
    N, C, OH, OW = win.shape[:4]
    return win.transpose(0, 1, 4, 5, 2, 3).reshape(N, C * KH * KW, OH * OW)


x = np.array([[[[1, 2, 0], [0, 1, 3], [2, 1, 0]]]], dtype=float)
w = np.array([[[[1, 0], [-1, 2]]]], dtype=float)
print(conv2d_np(x, w, np.zeros(1)))
print(im2col_np(x, 2, 2))
