import numpy as np
import cv2


def good_features(img, max_corners, quality, min_dist):
    ...


from skimage import data
print(good_features(data.camera(), 10, 0.01, 10))
