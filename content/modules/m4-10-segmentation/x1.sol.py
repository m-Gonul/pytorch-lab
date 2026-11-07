import numpy as np


def pixel_shuffle(x, r):
    Cr, H, W = x.shape
    C = Cr // (r * r)
    return x.reshape(C, r, r, H, W).transpose(0, 3, 1, 4, 2).reshape(C, H * r, W * r)


def pixel_unshuffle(y, r):
    C, Hr, Wr = y.shape
    H, W = Hr // r, Wr // r
    return y.reshape(C, H, r, W, r).transpose(0, 2, 4, 1, 3).reshape(C * r * r, H, W)


def subpixel_to_tconv(w, r):
    Cr, Ci = w.shape
    return w.reshape(Cr // (r * r), r, r, Ci).transpose(3, 0, 1, 2)
