import numpy as np
import cv2


def pipeline(img):
    ...


from skimage import data
r = pipeline(data.camera())
print({k: (v.dtype, round(float(v.mean()), 1)) for k, v in r.items()})
