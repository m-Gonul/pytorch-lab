import numpy as np


def _idx(n, r, mode):
    i = np.arange(-r, n + r)
    if mode == "replicate":
        return np.clip(i, 0, n - 1)
    if mode == "wrap":
        return np.mod(i, n)
    if mode == "reflect":
        p = 2 * n
        j = np.mod(i, p)
        return np.where(j < n, j, p - 1 - j)
    if mode == "reflect101":
        if n == 1:
            return np.zeros_like(i)
        p = 2 * n - 2
        j = np.mod(i, p)
        return np.where(j < n, j, p - j)
    raise ValueError(mode)


def pad(img, r, mode):
    H, W = img.shape
    if mode == "constant":
        out = np.zeros((H + 2 * r, W + 2 * r), img.dtype)
        out[r:r + H, r:r + W] = img
        return out
    return img[_idx(H, r, mode)][:, _idx(W, r, mode)]


a = np.arange(12).reshape(3, 4)
print(pad(a, 2, "reflect101"))
