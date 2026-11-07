import numpy as np


def intensity_slice(img, levels, colors):
    ...


def sin_pseudocolor(img, freq, phases):
    ...


from skimage import data
print(intensity_slice(data.camera(), [64, 128, 192], [(0, 0, 0), (255, 0, 0), (0, 255, 0), (255, 255, 255)]).shape)
