import numpy as np


def window_stats(img, win):
    ...


def min_dtype(H, W, bits=8):
    ...


from skimage import data
m, s = window_stats(data.camera(), 24)
print(m.shape, s.max())
