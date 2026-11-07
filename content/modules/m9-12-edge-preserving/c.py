import numpy as np
import cv2


def clahe_cv(img, clip=2.0, tiles=(8, 8)):
    ...


from skimage import data
print(clahe_cv(data.moon()).mean())
