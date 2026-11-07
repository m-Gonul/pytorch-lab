import numpy as np


def resize_bilinear(img, H2, W2):
    ...


def resize_cubic(img, H2, W2, a=-0.75):
    ...


from skimage import data
img = data.camera().astype(np.float32)
print(resize_bilinear(img, 200, 300).shape, resize_cubic(img, 700, 900).shape)
