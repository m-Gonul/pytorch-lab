import numpy as np


def dlt_raw(src, dst):
    ...


def dlt_norm(src, dst):
    ...


def cond_A(src, dst, normalize):
    ...


s = np.random.default_rng(0).uniform(0, 1000, (8, 2))
print(cond_A(s, s + 3, False), cond_A(s, s + 3, True))
