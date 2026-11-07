import numpy as np
import cv2


def _shift(f, dy, dx):
    p = np.pad(f, 1, mode="reflect")
    H, W = f.shape
    return p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]


def sharpen_laplacian(img, kind=4):
    f = img.astype(np.float64)
    L = _shift(f, -1, 0) + _shift(f, 1, 0) + _shift(f, 0, -1) + _shift(f, 0, 1) - 4 * f
    if kind == 8:
        L = L + _shift(f, -1, -1) + _shift(f, -1, 1) + _shift(f, 1, -1) + _shift(f, 1, 1) - 4 * f
    return np.clip(np.rint(f - L), 0, 255).astype(np.uint8)


def unsharp(img, k=1.0, ksize=5, sigma=0):
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (ksize, ksize), sigma)
    return np.clip(np.rint(f + k * (f - blur)), 0, 255).astype(np.uint8)


def gradient_magnitude(img, norm="l2"):
    f = img.astype(np.float32)
    s = lambda dy, dx: _shift(f, dy, dx)
    gx = (s(-1, 1) + 2 * s(0, 1) + s(1, 1)) - (s(-1, -1) + 2 * s(0, -1) + s(1, -1))
    gy = (s(1, -1) + 2 * s(1, 0) + s(1, 1)) - (s(-1, -1) + 2 * s(-1, 0) + s(-1, 1))
    return (np.sqrt(gx * gx + gy * gy) if norm == "l2" else np.abs(gx) + np.abs(gy)).astype(np.float32)


from skimage import data
img = data.moon()
print(sharpen_laplacian(img).dtype, unsharp(img, 2.0).mean(), gradient_magnitude(img).max())
