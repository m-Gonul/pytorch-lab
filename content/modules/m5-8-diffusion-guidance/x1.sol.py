import numpy as np


def linear_abar(T, b1=1e-4, bT=0.02):
    return np.cumprod(1 - np.linspace(b1, bT, T))


def cosine_abar(T, s=0.008):
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    beta = np.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    return np.cumprod(1 - beta)


def log_snr(abar):
    return np.log(abar / (1 - abar))


def wasted_fraction(abar, thr=0.01):
    return float((abar < thr).mean())
