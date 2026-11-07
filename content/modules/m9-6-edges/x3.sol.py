import numpy as np
from scipy import ndimage, signal


def gaussian_size(sigma):
    n = int(np.ceil(6 * sigma))
    return n + 1 - n % 2


def log_formula(sigma):
    n = gaussian_size(sigma)
    y, x = np.mgrid[-(n // 2):n // 2 + 1, -(n // 2):n // 2 + 1].astype(np.float64)
    r2 = x * x + y * y
    k = (r2 - 2 * sigma ** 2) / sigma ** 4 * np.exp(-r2 / (2 * sigma ** 2))
    return k - k.mean()


def log_via_laplacian(sigma):
    n = gaussian_size(sigma)
    t = np.arange(n) - n // 2
    g = np.exp(-t ** 2 / (2 * sigma ** 2))
    G = np.outer(g, g)
    G /= G.sum()
    L = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float64)
    return signal.convolve2d(G, L, mode="full")


def derivative_noise(noise_var, smooth=None, seed=0):
    x = np.arange(256)
    f = np.clip((x - 96) / 64, 0, 1) * 200
    f = f + np.random.default_rng(seed).normal(0, np.sqrt(noise_var), 256)
    if smooth:
        f = ndimage.gaussian_filter1d(f, smooth)
    return np.diff(f)[10:80].std(), np.diff(f, 2)[10:80].std()


print(gaussian_size(1.4), log_formula(1.4).sum())
