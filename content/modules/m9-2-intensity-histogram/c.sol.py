import numpy as np


def equalize(img, variant="opencv"):
    h = np.bincount(img.ravel(), minlength=256)
    c = h.cumsum()
    N = img.size
    if variant == "class":
        lut = np.rint(255 * c / N)
    else:
        cmin = h[np.flatnonzero(h)[0]]
        if N == cmin:
            return img.copy()
        lut = np.clip(np.rint(255 * (c - cmin) / (N - cmin)), 0, 255)
    return lut.astype(np.uint8)[img]


def negative(img):
    return (255 - img).astype(np.uint8)


def log_transform(img):
    lut = np.rint(255 / np.log(256) * np.log1p(np.arange(256)))
    return lut.astype(np.uint8)[img]


def gamma(img, g):
    lut = np.rint(255 * (np.arange(256) / 255) ** g)
    return lut.astype(np.uint8)[img]


from skimage import data
img = data.moon()
print(equalize(img).min(), equalize(img).max(), gamma(img, 0.5).mean())
