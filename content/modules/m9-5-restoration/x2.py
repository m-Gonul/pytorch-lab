import numpy as np


def geometric_mean(img, k):
    ...


def harmonic_mean(img, k):
    ...


def contraharmonic(img, k, Q):
    ...


from skimage import data
print(geometric_mean(data.camera(), 3).mean())
