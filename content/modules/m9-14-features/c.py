import numpy as np


def hog(img, nbins=9, cell=8, block=2):
    ...


def fast_hessian(img, L=9):
    ...


from skimage import data
print(hog(data.camera()[:128, :64]).shape)
