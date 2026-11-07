import numpy as np


def absorption(P, absorbing):
    n = len(P)
    tr = [i for i in range(n) if i not in absorbing]
    Q = P[np.ix_(tr, tr)]
    Rm = P[np.ix_(tr, list(absorbing))]
    Nf = np.linalg.inv(np.eye(len(tr)) - Q)
    return Nf @ Rm, Nf @ np.ones(len(tr))


def gamblers_ruin(N, p):
    P = np.zeros((N + 1, N + 1))
    P[0, 0] = P[N, N] = 1
    for i in range(1, N):
        P[i, i + 1] = p
        P[i, i - 1] = 1 - p
    B, _ = absorption(P, [0, N])
    return B[:, 1]


def mean_first_passage(P):
    n = len(P)
    M = np.zeros((n, n))
    for j in range(n):
        o = [k for k in range(n) if k != j]
        m = np.linalg.solve(np.eye(n - 1) - P[np.ix_(o, o)], np.ones(n - 1))
        M[o, j] = m
        M[j, j] = 1 + P[j, o] @ m
    return M


print(gamblers_ruin(4, 0.5))
