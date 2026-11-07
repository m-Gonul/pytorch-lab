import os
import time
import hashlib
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


def mandel_torch(H, W, maxit):
    cr = (-2.0 + 3.0 * torch.arange(W, dtype=torch.float64) / (W - 1))[None].expand(H, W)
    ci = (1.5 * torch.arange(H, dtype=torch.float64) / (H - 1))[:, None].expand(H, W)
    zr = torch.zeros(H, W, dtype=torch.float64)
    zi = torch.zeros_like(zr)
    k = torch.zeros(H, W, dtype=torch.int32)
    for _ in range(maxit):
        alive = zr * zr + zi * zi <= 4.0
        if not bool(alive.any()):
            break
        t = zr * zr - zi * zi + cr
        zi = torch.where(alive, 2 * zr * zi + ci, zi)
        zr = torch.where(alive, t, zr)
        k += alive.int()
    return k


def scaling(ext, H, W, maxit, threads):
    times = []
    for p in threads:
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            ext.mandelbrot(H, W, maxit, p, "dynamic")
            best = min(best, time.perf_counter() - t0)
        times.append(best)
    return np.array(times), times[0] / np.array(times)


def fit_amdahl(p, T):
    p = np.asarray(p, float)
    X = np.stack([np.ones_like(p), 1.0 / p], 1)
    a, b = np.linalg.lstsq(X, np.asarray(T, float), rcond=None)[0]
    return a / (a + b), a + b


print(mandel_torch(50, 75, 100).float().mean())
