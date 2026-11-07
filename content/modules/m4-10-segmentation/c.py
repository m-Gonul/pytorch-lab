import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv_matrix(w, H, s=1, p=0):
    ...


def tconv2d(x, w, s=1, p=0):
    ...


def tconv_zero_insert(x, w, s=1, p=0):
    ...


def bilinear_kernel(f):
    ...


def confusion(pred, gt, C):
    ...


def miou(cm):
    ...
