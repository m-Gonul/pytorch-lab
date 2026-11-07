import numpy as np


def _win(img, k):
    r = k // 2
    return np.lib.stride_tricks.sliding_window_view(np.pad(img.astype(np.float64) + 1, r, mode="edge"), (k, k))


def geometric_mean(img, k):
    return np.exp(np.log(_win(img, k)).mean((-2, -1))) - 1


def harmonic_mean(img, k):
    return k * k / (1 / _win(img, k)).sum((-2, -1)) - 1


def contraharmonic(img, k, Q):
    w = _win(img, k)
    return (w ** (Q + 1)).sum((-2, -1)) / (w ** Q).sum((-2, -1)) - 1


from skimage import data
print(geometric_mean(data.camera(), 3).mean())
