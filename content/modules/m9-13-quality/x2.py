import numpy as np
from scipy import ndimage


def rgb_to_y(img):
    ...


def psnr_y(a, b, shave=0):
    ...


def ssim_rgb(a, b):
    ...


from skimage import data
print(rgb_to_y(data.astronaut()).min())
