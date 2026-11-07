import numpy as np


def sliding_windows(x, w):
    return np.lib.stride_tricks.sliding_window_view(x, w)


def moving_average(x, w):
    return sliding_windows(x, w).mean(axis=1)


def is_view_of(a, base):
    return bool(np.shares_memory(a, base))


x = np.arange(6.0)
W = sliding_windows(x, 3)
print(W, W.strides, is_view_of(W, x))
print(moving_average(x, 3))          # [1. 2. 3. 4.]
