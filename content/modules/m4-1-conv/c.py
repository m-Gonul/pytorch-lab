import numpy as np


def conv2d_np(x, w, b, stride=1, pad=0):
    """x: (N, C, H, W), w: (F, C, KH, KW) → (N, F, OH, OW). Döngü yok."""
    ...


def im2col_np(x, KH, KW, stride=1, pad=0):
    """→ (N, C·KH·KW, OH·OW). Döngü yok."""
    ...


x = np.array([[[[1, 2, 0], [0, 1, 3], [2, 1, 0]]]], dtype=float)
w = np.array([[[[1, 0], [-1, 2]]]], dtype=float)
print(conv2d_np(x, w, np.zeros(1)))
print(im2col_np(x, 2, 2))
