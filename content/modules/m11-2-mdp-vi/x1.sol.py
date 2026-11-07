import numpy as np


def vacuum_mdp():
    S, A = 8, 4
    P = np.zeros((A, S, S))
    R = np.zeros((S, A))
    for s in range(S):
        L, K, low = s & 1, (s >> 1) & 1, (s >> 2) & 1
        bonus = 20.0 if (L == 0 and K == 0) else 0.0
        for a in range(A):
            outs = []
            if a in (0, 1):
                dirty = L if a == 0 else K
                for clean_ok, pc in ((1, 0.9), (0, 0.1)):
                    nl, nk = L, K
                    gain = 0.0
                    if dirty and clean_ok:
                        if a == 0:
                            nl = 0
                        else:
                            nk = 0
                        gain = 10.0
                    if low:
                        outs.append((0.5 * pc, L, K, 0, -1.0 - 50.0))
                        outs.append((0.5 * pc, nl, nk, 1, gain - 1.0))
                    else:
                        outs.append((0.3 * pc, nl, nk, 1, gain - 1.0))
                        outs.append((0.7 * pc, nl, nk, 0, gain - 1.0))
            elif a == 2:
                outs.append((1.0, L, K, 0, 0.0))
            else:
                outs.append((1.0, L, K, low, 0.0))
            for p, nl, nk, nb, r in outs:
                R[s, a] += p * (r + bonus)
                for dl, pl in ((1, 0.1), (0, 0.9)):
                    for dk, pk in ((1, 0.1), (0, 0.9)):
                        P[a, s, (nl | dl) + 2 * (nk | dk) + 4 * nb] += p * pl * pk
    return P, R


def solve(gamma=0.95):
    P, R = vacuum_mdp()
    V = np.zeros(8)
    for _ in range(2000):
        V = (R + gamma * np.einsum("ast,t->sa", P, V)).max(1)
    return V, (R + gamma * np.einsum("ast,t->sa", P, V)).argmax(1)


print(solve()[1])
