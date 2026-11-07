import numpy as np


def stretch(img):
    f = img.astype(np.float64)
    lo, hi = f.min(), f.max()
    if hi == lo:
        return np.zeros_like(img)
    return np.rint((f - lo) * 255 / (hi - lo)).astype(np.uint8)


def piecewise(img, r1, s1, r2, s2):
    lut = np.interp(np.arange(256), [0, r1, r2, 255], [0, s1, s2, 255])
    return np.rint(lut).astype(np.uint8)[img]


def slice_level(img, lo, hi, mode="binary"):
    inside = (img >= lo) & (img <= hi)
    if mode == "binary":
        return np.where(inside, 255, 0).astype(np.uint8)
    return np.where(inside, 255, img).astype(np.uint8)


from skimage import data
img = data.moon()
print(stretch(img).min(), stretch(img).max())
