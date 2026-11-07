import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def dilated_conv2d(x, w, d):
    k = w.shape[0]
    r = d * (k - 1) // 2
    xp = np.pad(x, r)
    win = sliding_window_view(xp, (d * (k - 1) + 1,) * 2)[:, :, ::d, ::d]
    return np.einsum("hwuv,uv->hw", win, w)


def receptive_field(ks, ds):
    return 1 + sum((k - 1) * d for k, d in zip(ks, ds))


def footprint(ds, k=3):
    R = receptive_field([k] * len(ds), ds)
    c = R // 2
    F = np.zeros((R, R), bool)
    F[c, c] = True
    off = np.arange(k) - k // 2
    for d in ds:
        G = np.zeros_like(F)
        for a in off:
            for b in off:
                G |= np.roll(np.roll(F, a * d, 0), b * d, 1)
        F = G
    return F
