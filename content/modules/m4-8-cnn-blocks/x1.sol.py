import numpy as np


def gap_head(F, W, b):
    return W @ F.mean((1, 2)) + b


def cam(F, W, c):
    return np.einsum("k,khw->hw", W[c], F)


def params_flatten_vs_gap(K, H, W, classes, hidden):
    return K * H * W * hidden + hidden + hidden * classes + classes, K * classes + classes
