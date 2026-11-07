import numpy as np
import cv2


def gaussian_pyramid(img, levels):
    ...


def laplacian_pyramid(img, levels):
    ...


def collapse(L):
    ...


def blend(a, b, mask, levels=5):
    ...


from skimage import data
x = data.camera().astype(np.float64)
print(np.abs(collapse(laplacian_pyramid(x, 5)) - x).max())
