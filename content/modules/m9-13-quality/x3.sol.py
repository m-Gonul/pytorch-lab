import math
import numpy as np
import cv2
from scipy import signal

N = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)


def estimate_noise(img):
    H, W = img.shape
    s = np.abs(signal.convolve2d(img.astype(np.float64), N, mode="valid")).sum()
    return math.sqrt(math.pi / 2) * s / (6 * (W - 2) * (H - 2))


def blur_score(img):
    return float(cv2.Laplacian(img.astype(np.float64), cv2.CV_64F).var())


from skimage import data
print(estimate_noise(data.camera()))
