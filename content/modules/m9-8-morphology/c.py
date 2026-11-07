import numpy as np


def gray_erode(f, se):
    ...


def gray_dilate(f, se):
    ...


def opening(f, se):
    ...


def closing(f, se):
    ...


def gradient(f, se):
    ...


def tophat(f, se):
    ...


def blackhat(f, se):
    ...


def nonflat_erode(f, bN):
    ...


from skimage import data
print(gradient(data.camera(), np.ones((3, 3), bool)).mean())
