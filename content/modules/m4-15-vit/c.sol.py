import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def patchify(x, P):
    C, H, W = x.shape
    return x.reshape(C, H // P, P, W // P, P).transpose(1, 3, 2, 4, 0).reshape((H // P) * (W // P), P * P * C)


def unpatchify(z, P, C, H, W):
    return z.reshape(H // P, W // P, P, P, C).transpose(4, 0, 2, 1, 3).reshape(C, H, W)


def patch_embed_conv(x, Wc, b, P):
    win = sliding_window_view(x, (P, P), axis=(1, 2))[:, ::P, ::P]
    out = np.einsum("chwuv,dcuv->hwd", win, Wc) + b
    return out.reshape(-1, Wc.shape[0])


def conv_to_linear(Wc):
    D = Wc.shape[0]
    return Wc.transpose(2, 3, 1, 0).reshape(-1, D)


def mean_attention_distance(A, g, P):
    yy, xx = np.divmod(np.arange(g * g), g)
    dist = np.hypot(yy[:, None] - yy[None], xx[:, None] - xx[None]) * P
    return float((A * dist).sum(1).mean())
