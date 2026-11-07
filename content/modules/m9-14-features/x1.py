import numpy as np
import cv2


def box_sum(ii, y0, x0, h, w):
    ...


def haar(ii, y, x, s):
    ...


def usurf(img, pts, s):
    ...


from skimage import data
print(usurf(data.camera(), np.array([[256, 256]]), 2).shape)
