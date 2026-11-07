import numpy as np


def critical_path(A, dur):
    n = len(dur)
    ef = dur.astype(float).copy()
    M = np.where(A > 0, 0.0, -np.inf)
    for _ in range(n):
        ef = dur + np.maximum((M + ef[:, None]).max(0), 0.0)
    return ef, float(ef.max())


def fit_amdahl(p, T):
    X = np.stack([np.ones_like(p, dtype=float), 1.0 / p], 1)
    a, b = np.linalg.lstsq(X, T, rcond=None)[0]
    return a / (a + b), a + b


def speedup_grid(f, p):
    f, p = f[:, None], p[None, :]
    return 1 / ((1 - f) + f / p), (1 - f) + f * p


A = np.zeros((4, 4)); A[0, 1] = A[0, 2] = A[1, 3] = A[2, 3] = 1
print(critical_path(A, np.array([2.0, 3, 1, 4])))
