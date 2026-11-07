import numpy as np


def bit_planes(img):
    return ((img[None] >> np.arange(8, dtype=np.uint8)[:, None, None]) & 1).astype(np.uint8)


def requantize(img, bits):
    step = 2 ** (8 - bits)
    return ((img // step) * step).astype(np.uint8)


from skimage import data
img = data.camera()
print(bit_planes(img).shape, np.unique(requantize(img, 2)))
