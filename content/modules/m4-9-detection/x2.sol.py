import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def fpn_level(w, h, k0=4, kmin=2, kmax=5):
    k = np.floor(k0 + np.log2(np.sqrt(np.asarray(w, float) * np.asarray(h, float)) / 224))
    return np.clip(k, kmin, kmax).astype(int)


def upsample2(x):
    return x.repeat(2, -2).repeat(2, -1)


def _conv3(x, w):
    win = sliding_window_view(np.pad(x, ((0, 0), (1, 1), (1, 1))), (3, 3), axis=(1, 2))
    return np.einsum("chwuv,ocuv->ohw", win, w)


def top_down(feats, lat_w, smooth_w):
    M = [None] * len(feats)
    M[-1] = np.einsum("dc,chw->dhw", lat_w[-1], feats[-1])
    for i in range(len(feats) - 2, -1, -1):
        M[i] = upsample2(M[i + 1]) + np.einsum("dc,chw->dhw", lat_w[i], feats[i])
    return [_conv3(m, w) for m, w in zip(M, smooth_w)]
