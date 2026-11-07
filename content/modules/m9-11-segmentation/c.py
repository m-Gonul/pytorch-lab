import numpy as np


def kmeans_np(Z, init, max_iter=300):
    ...


def quantize(img, k, seed=0):
    ...


from skimage import data
out, lab = quantize(data.astronaut()[::4, ::4], 8)
print(out.shape, lab.max())
