import numpy as np
import cv2


def sharpen_laplacian(img, kind=4):
    ...


def unsharp(img, k=1.0, ksize=5, sigma=0):
    ...


def gradient_magnitude(img, norm="l2"):
    ...


from skimage import data
img = data.moon()
print(sharpen_laplacian(img).dtype, unsharp(img, 2.0).mean(), gradient_magnitude(img).max())
