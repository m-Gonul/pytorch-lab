import math
import numpy as np
import cv2


def bilateral_color(img, d, sc, ss):
    ...


def joint_bilateral(img, guide, d, sc, ss):
    ...


from skimage import data
print(bilateral_color(data.astronaut()[:64, :64], 7, 30, 5).shape)
