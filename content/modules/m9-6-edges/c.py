import numpy as np
import cv2
from scipy import ndimage


def canny_cv(img, lo, hi, l2=False):
    ...


def marr_hildreth(img, sigma, thr_frac=0.04):
    ...


from skimage import data
print((canny_cv(data.camera(), 50, 150) > 0).sum())
