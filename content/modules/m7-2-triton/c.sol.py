import numpy as np


def load_tile(M, r0, c0, R, Cc):
    t = np.zeros((R, Cc))
    sub = M[r0:r0 + R, c0:c0 + Cc]
    t[:sub.shape[0], :sub.shape[1]] = sub
    return t


def matmul_program(A, B, C, pid_m, pid_n, BM, BN, BK):
    r0, c0 = pid_m * BM, pid_n * BN
    acc = np.zeros((BM, BN))
    for k in range(0, A.shape[1], BK):
        acc += load_tile(A, r0, k, BM, BK) @ load_tile(B, k, c0, BK, BN)
    rr = min(BM, C.shape[0] - r0)
    cc = min(BN, C.shape[1] - c0)
    C[r0:r0 + rr, c0:c0 + cc] = acc[:rr, :cc]


def matmul(A, B, BM=16, BN=16, BK=16):
    M, N = A.shape[0], B.shape[1]
    C = np.empty((M, N))
    for pm in range(-(-M // BM)):
        for pn in range(-(-N // BN)):
            matmul_program(A, B, C, pm, pn, BM, BN, BK)
    return C


rng = np.random.default_rng(0)
A, B = rng.normal(size=(70, 50)), rng.normal(size=(50, 40))
print(np.abs(matmul(A, B) - A @ B).max())
