import numpy as np


def load_tile(M, r0, c0, R, Cc):
    ...


def matmul_program(A, B, C, pid_m, pid_n, BM, BN, BK):
    ...


def matmul(A, B, BM=16, BN=16, BK=16):
    ...


rng = np.random.default_rng(0)
A, B = rng.normal(size=(70, 50)), rng.normal(size=(50, 40))
print(np.abs(matmul(A, B) - A @ B).max())
