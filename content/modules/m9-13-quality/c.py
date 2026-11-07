import numpy as np
import cv2
from scipy import ndimage


def ssim(x, y, R=255):
    ...


def mean_shift(img, mse):
    ...


def contrast(img, mse):
    ...


def noise(img, mse, seed=0):
    ...


def impulse(img, mse, seed=0):
    ...


def blur(img, mse):
    ...


def jpeg(img, mse):
    ...


from skimage import data
print(ssim(data.camera(), mean_shift(data.camera(), 210)))
