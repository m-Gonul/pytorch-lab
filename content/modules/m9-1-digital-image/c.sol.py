import numpy as np


def _axis_linear(n, n2):
    x = np.clip((np.arange(n2) + 0.5) * n / n2 - 0.5, 0, n - 1)
    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, n - 1)
    return x0, x1, x - x0


def resize_bilinear(img, H2, W2):
    H, W = img.shape
    y0, y1, wy = _axis_linear(H, H2)
    x0, x1, wx = _axis_linear(W, W2)
    wy, wx = wy[:, None], wx[None, :]
    a, b = img[y0][:, x0], img[y0][:, x1]
    c, d = img[y1][:, x0], img[y1][:, x1]
    return ((1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * c + wx * d)).astype(np.float32)


def _keys(t, a):
    t = np.abs(t)
    return np.where(t <= 1, (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1,
                    np.where(t < 2, a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a, 0.0))


def _axis_cubic(n, n2, a):
    x = (np.arange(n2) + 0.5) * n / n2 - 0.5
    x0 = np.floor(x).astype(int)
    off = np.arange(-1, 3)
    idx = np.clip(x0[:, None] + off[None, :], 0, n - 1)
    w = _keys((x - x0)[:, None] - off[None, :], a)
    return idx, w


def resize_cubic(img, H2, W2, a=-0.75):
    H, W = img.shape
    iy, wy = _axis_cubic(H, H2, a)
    ix, wx = _axis_cubic(W, W2, a)
    rows = (img[iy] * wy[:, :, None]).sum(1)
    return (rows[:, ix] * wx[None]).sum(2).astype(np.float32)


from skimage import data
img = data.camera().astype(np.float32)
print(resize_bilinear(img, 200, 300).shape, resize_cubic(img, 700, 900).shape)
