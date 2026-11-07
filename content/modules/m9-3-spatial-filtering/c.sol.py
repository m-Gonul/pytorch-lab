import numpy as np


def _box1d(a, size, axis):
    r = size // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    cs = np.cumsum(np.pad(a, pad), axis=axis)
    n = a.shape[axis]
    hi = np.take(cs, np.arange(size, size + n), axis=axis)
    lo = np.take(cs, np.arange(0, n), axis=axis)
    return hi - lo


def box_separable(img, size):
    f = img.astype(np.float64)
    return (_box1d(_box1d(f, size, 1), size, 0) / (size * size)).astype(np.float32)


def integral(img):
    A = np.zeros((img.shape[0] + 1, img.shape[1] + 1), np.int64)
    A[1:, 1:] = img.astype(np.int64).cumsum(0).cumsum(1)
    return A


def box_integral(img, size):
    r = size // 2
    A = integral(np.pad(img, r))
    H, W = img.shape
    s = A[size:size + H, size:size + W] - A[0:H, size:size + W] - A[size:size + H, 0:W] + A[0:H, 0:W]
    return (s / (size * size)).astype(np.float32)


SMALL = {1: [1.0], 3: [0.25, 0.5, 0.25], 5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
         7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]}


def gaussian_kernel(ksize, sigma=0):
    if sigma <= 0 and ksize in SMALL:
        return np.array(SMALL[ksize])
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    x = np.arange(ksize) - (ksize - 1) / 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_blur(img, ksize, sigma=0):
    g = gaussian_kernel(ksize, sigma)
    r = ksize // 2
    f = np.pad(img.astype(np.float64), r, mode="reflect")
    H, W = img.shape
    idx = np.arange(ksize)
    rows = (f[:, np.arange(W)[:, None] + idx] * g).sum(-1)
    out = (rows[np.arange(H)[:, None] + idx, :] * g[None, :, None]).sum(1)
    return out.astype(np.float32)


from skimage import data
img = data.camera()
print(integral(img)[-1, -1], box_integral(img, 21).shape, gaussian_kernel(7).round(4))
