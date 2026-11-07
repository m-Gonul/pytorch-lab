import numpy as np
import cv2


def match_ratio(d1, d2, ratio=0.8, cross=False):
    ...


def evaluate(img, M, detector):
    ...


from skimage import data
print(evaluate(data.camera(), cv2.getRotationMatrix2D((256, 256), 20, 0.8), "sift"))
