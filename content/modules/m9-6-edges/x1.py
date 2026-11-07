import numpy as np
import cv2


def kirsch(img):
    ...


from skimage import data
r, i = kirsch(data.camera())
print(r.max(), np.bincount(i.ravel(), minlength=8))
