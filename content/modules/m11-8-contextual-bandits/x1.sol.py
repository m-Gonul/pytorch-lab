import time
import numpy as np


def sm_update(Ainv, x):
    Ax = Ainv @ x
    return Ainv - np.outer(Ax, Ax) / (1 + x @ Ax)


def time_compare(d, n_updates=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_updates, d))
    A = np.eye(d)
    t0 = time.perf_counter()
    for x in X:
        A = A + np.outer(x, x)
        inv = np.linalg.inv(A)
    t1 = time.perf_counter()
    S = np.eye(d)
    for x in X:
        S = sm_update(S, x)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, float(np.abs(S - inv).max())


print(time_compare(100))
