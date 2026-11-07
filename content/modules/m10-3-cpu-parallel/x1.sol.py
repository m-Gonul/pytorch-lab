import numpy as np


def separable_error(K, sigma):
    x = np.arange(K) - K // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    G = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    G /= G.sum()
    return float(np.abs(np.outer(g, g) - G).max())


def svd_rank_approx(kernel, r):
    U, S, Vt = np.linalg.svd(kernel)
    approx = (U[:, :r] * S[:r]) @ Vt[:r]
    return approx, float(np.linalg.norm(kernel - approx) / np.linalg.norm(kernel))


def cost(K, r):
    return 2 * K * r, K * K


print(separable_error(21, 3.0))
