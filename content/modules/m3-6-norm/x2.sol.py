import numpy as np


def fold_linear(W, b, gamma, beta, mean, var, eps=1e-5):
    s = gamma / np.sqrt(var + eps)
    return W * s[:, None], (b - mean) * s + beta


def fold_conv(W, b, gamma, beta, mean, var, eps=1e-5):
    s = gamma / np.sqrt(var + eps)
    return W * s[:, None, None, None], (b - mean) * s + beta
