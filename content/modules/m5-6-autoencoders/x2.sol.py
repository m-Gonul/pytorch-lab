import numpy as np


def encoder_jacobian(x, W, b):
    h = 1 / (1 + np.exp(-(W @ x + b)))
    return (h * (1 - h))[:, None] * W


def cae_penalty(X, W, b):
    H = 1 / (1 + np.exp(-(X @ W.T + b)))
    return float(((H * (1 - H)) ** 2 @ (W ** 2).sum(1)).mean())
