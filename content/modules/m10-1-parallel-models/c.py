import numpy as np


def critical_path(A, dur):
    ...


def fit_amdahl(p, T):
    ...


def speedup_grid(f, p):
    ...


A = np.zeros((4, 4)); A[0, 1] = A[0, 2] = A[1, 3] = A[2, 3] = 1
print(critical_path(A, np.array([2.0, 3, 1, 4])))
