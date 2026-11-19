import math
import numpy as np


def average_evaluate(P, r, anchor=0):
    P, r = np.asarray(P, dtype=float), np.asarray(r, dtype=float)
    n = len(r)
    A = np.column_stack((np.eye(n) - P, np.ones(n)))
    constraint = np.zeros(n + 1)
    constraint[anchor] = 1.0
    solution = np.linalg.solve(np.vstack((A, constraint)), np.r_[r, 0.0])
    return float(solution[-1]), solution[:-1]


def access_kernel(p=0.06, release_new=True):
    D = np.zeros((11, 2, 11))
    for free in range(11):
        for action in range(2):
            accepted = min(action, free)
            after = free - accepted
            busy = 10 - (after if release_new else free)
            for k in range(busy + 1):
                D[free, action, after + k] = math.comb(busy, k) * p**k * (1-p)**(busy-k)
    return D


def optimal_access(p=0.06, epsilon=0.0, release_new=True):
    D = access_kernel(p, release_new)
    reward = np.array([1.0, 2.0, 4.0, 8.0])
    accept = np.full((11, 4), epsilon / 2)
    accept[0] = 0.0
    for _ in range(100):
        policy = np.stack((1-accept, accept), axis=-1)
        P = np.einsum("fpa,fak->fk", policy, D) / 4
        r = accept @ reward / 4
        rho, h = average_evaluate(P, r)
        advantage = reward[None, :] + ((D[:, 1] - D[:, 0]) @ h)[:, None]
        greedy = (advantage > 0).astype(int)
        greedy[0] = 0
        updated = epsilon / 2 + (1-epsilon) * greedy
        updated[0] = 0.0
        if np.array_equal(updated, accept):
            return dict(rho=rho, h=h, accept=accept, greedy=greedy, P=P, r=r)
        accept = updated
    raise RuntimeError("Politika yinelemesi100 turda durmadı")


print(optimal_access()["rho"])
