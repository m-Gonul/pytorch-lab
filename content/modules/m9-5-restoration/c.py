import numpy as np
import cv2


def median_np(img, k):
    ...


def adaptive_mean(img_u8, noise_var, k=5):
    ...


def psnr(a, b):
    ...


from skimage import data
img = data.camera()
print(psnr(img, median_np(img, 5)))
