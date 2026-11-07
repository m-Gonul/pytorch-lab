import numpy as np
import cv2


def granulometry(img, radii):
    ...


def pattern_spectrum(img, radii):
    ...


def oc_smooth(img, r):
    ...


from skimage import data
print(pattern_spectrum(data.coins(), range(1, 30, 2)).argmax())
