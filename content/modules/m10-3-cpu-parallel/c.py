import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def scan_blocks(a, p):
    ...


def integral_image(img):
    ...


def conv2d(img, k):
    ...


def conv2d_separable(img, kx, ky):
    ...


def matmul_tiled(A, B, T):
    ...


print(scan_blocks(np.arange(1, 9), 4))
