import numpy as np
import cv2


def local_enhance(img, E=4.0, k0=0.4, k1=0.02, k2=0.4, size=3):
    ...


from skimage import data
print(local_enhance(data.camera()).mean())
