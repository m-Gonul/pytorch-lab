import numpy as np


def separate(k, tol=1e-8):
    ...


import cv2
g = cv2.getGaussianKernel(5, 0) @ cv2.getGaussianKernel(5, 0).T
print(separate(g))
