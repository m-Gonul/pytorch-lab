import numpy as np
from scipy import ndimage

K = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]]) / 12.0


def horn_schunck(I0, I1, alpha=10.0, iters=200):
    I0, I1 = I0.astype(np.float64), I1.astype(np.float64)
    m = (I0 + I1) / 2
    Iy, Ix = np.gradient(m)
    It = I1 - I0
    u = np.zeros_like(I0); v = np.zeros_like(I0)
    den = alpha ** 2 + Ix ** 2 + Iy ** 2
    for _ in range(iters):
        ub, vb = ndimage.convolve(u, K, mode="nearest"), ndimage.convolve(v, K, mode="nearest")
        t = (Ix * ub + Iy * vb + It) / den
        u, v = ub - Ix * t, vb - Iy * t
    return u, v


print("Horn–Schunck")
