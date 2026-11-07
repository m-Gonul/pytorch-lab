import numpy as np


def rgb2hsv8(img):
    ...


def rgb2ycrcb8(img):
    ...


def rgb2hsi(img):
    ...


from skimage import data
print(rgb2hsv8(data.astronaut())[0, 0])
