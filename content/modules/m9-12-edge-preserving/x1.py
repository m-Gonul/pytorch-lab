import numpy as np
import cv2


def ahe_sliding(img, w):
    ...


def noise_gain(img, out, region):
    ...


from skimage import data
print(ahe_sliding(data.camera()[:64, :64], 15).mean())
