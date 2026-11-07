import numpy as np


def bit_planes(img):
    ...


def requantize(img, bits):
    ...


from skimage import data
img = data.camera()
print(bit_planes(img).shape, np.unique(requantize(img, 2)))
