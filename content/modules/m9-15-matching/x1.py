import numpy as np
import cv2


def best_match(img, t, method):
    ...


def multiscale(img, t, scales):
    ...


from skimage import data
im = data.camera().astype(np.float32)
print(best_match(im, im[100:140, 200:250].copy(), cv2.TM_CCOEFF_NORMED))
