import numpy as np


def sliding_windows(x, w):
    ...


def moving_average(x, w):
    ...


def is_view_of(a, base):
    ...


x = np.arange(6.0)
W = sliding_windows(x, 3)
print(W, W.strides, is_view_of(W, x))
print(moving_average(x, 3))          # [1. 2. 3. 4.]
