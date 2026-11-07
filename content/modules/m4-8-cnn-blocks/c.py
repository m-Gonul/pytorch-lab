import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv2d(x, w):
    ...


def depthwise(x, w):
    ...


def pointwise(x, w):
    ...


def grouped_conv(x, w, g):
    ...


def se_block(x, W1, b1, W2, b2):
    ...


def lrn(x, k=2, n=5, alpha=1e-4, beta=0.75):
    ...
