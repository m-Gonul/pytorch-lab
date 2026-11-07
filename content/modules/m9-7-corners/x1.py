import numpy as np
import cv2


def eigen_maps(img, sigma=1.5):
    ...


def harris_from_eigs(l1, l2, k):
    ...


def count_corners(R, T):
    ...


from skimage import data
l1, l2 = eigen_maps(data.checkerboard() / 255.0)
print(l1.max(), l2.max())
