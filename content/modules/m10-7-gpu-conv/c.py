import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def im2col(img, K):
    ...


def conv_im2col(img, k):
    ...


def conv_multi(x, w):
    ...


def conv_fft(img, k):
    ...


def tile_intensity(BM, BN, BK):
    ...


print(im2col(np.arange(9.0).reshape(3, 3), 3).shape)
