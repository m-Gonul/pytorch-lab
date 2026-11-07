import numpy as np
import cv2
from scipy import ndimage


def corner_harris_cv(img, block, ksize, k):
    ...


def corner_min_eig_cv(img, block, ksize):
    ...


def harris_points(img, k=0.04, T_frac=0.01, r=3):
    ...


from skimage import data
print(harris_points(data.camera())[:5])
