import numpy as np
import cv2


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def granulometry(img, radii):
    return np.array([cv2.morphologyEx(img, cv2.MORPH_OPEN, _disk(r)).astype(np.float64).sum() for r in radii])


def pattern_spectrum(img, radii):
    g = granulometry(img, radii)
    d = -np.diff(g)
    return d / d.sum()


def oc_smooth(img, r):
    k = _disk(r)
    return cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


from skimage import data
print(pattern_spectrum(data.coins(), range(1, 30, 2)).argmax())
