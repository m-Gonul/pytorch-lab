import numpy as np


def rotate_bilinear(img, deg):
    ...


def estimate_affine(src, dst):
    ...


from skimage import data
img = data.camera().astype(np.float64)
print(rotate_bilinear(img, 30).shape)
