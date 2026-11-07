import numpy as np
import cv2
from scipy import ndimage


def _abc(img, block, ksize):
    s = 1.0 / ((1 << (ksize - 1)) * block * 255.0)
    f = img.astype(np.float32)
    dx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=ksize, scale=s)
    dy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=ksize, scale=s)
    box = lambda a: cv2.boxFilter(a, -1, (block, block), normalize=False)
    return box(dx * dx), box(dx * dy), box(dy * dy)


def corner_harris_cv(img, block, ksize, k):
    a, b, c = _abc(img, block, ksize)
    return a * c - b * b - np.float32(k) * (a + c) ** 2


def corner_min_eig_cv(img, block, ksize):
    a, b, c = _abc(img, block, ksize)
    a, c = a * 0.5, c * 0.5
    return (a + c) - np.sqrt((a - c) ** 2 + b * b)


def harris_points(img, k=0.04, T_frac=0.01, r=3):
    R = corner_harris_cv(img, 3, 3, k)
    mx = ndimage.maximum_filter(R, 2 * r + 1, mode="constant", cval=-np.inf)
    ys, xs = np.nonzero((R > T_frac * R.max()) & (R >= mx))
    o = np.argsort(-R[ys, xs], kind="stable")
    return np.stack([xs[o], ys[o]], 1)


from skimage import data
print(harris_points(data.camera())[:5])
