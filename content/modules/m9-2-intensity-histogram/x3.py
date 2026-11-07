import numpy as np


def stretch(img):
    ...


def piecewise(img, r1, s1, r2, s2):
    ...


def slice_level(img, lo, hi, mode="binary"):
    ...


from skimage import data
img = data.moon()
print(stretch(img).min(), stretch(img).max())
