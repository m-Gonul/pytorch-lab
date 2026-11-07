import numpy as np
from scipy import signal


def ncc_map(img, t):
    ...


def normalize_points(p):
    ...


def dlt(src, dst):
    ...


def ransac(src, dst, thr=3.0, iters=500, seed=0):
    ...


rng = np.random.default_rng(0)
s = rng.uniform(0, 100, (10, 2))
print(dlt(s, s + 5))
