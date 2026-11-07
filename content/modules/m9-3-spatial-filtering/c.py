import numpy as np


def box_separable(img, size):
    ...


def integral(img):
    ...


def box_integral(img, size):
    ...


def gaussian_kernel(ksize, sigma=0):
    ...


def gaussian_blur(img, ksize, sigma=0):
    ...


from skimage import data
img = data.camera()
print(integral(img)[-1, -1], box_integral(img, 21).shape, gaussian_kernel(7).round(4))
