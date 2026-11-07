import numpy as np
import torch
from scipy.stats import poisson


def inventory_mdp(C=50, max_order=20, lam=8.0, price=10.0, cost=4.0, hold=0.5, short=2.0):
    ...


def vi_vectorized(P, R, gamma, eps=1e-8):
    ...


def vi_loop(P, R, gamma, eps=1e-8):
    A, S, _ = P.shape
    V = np.zeros(S)
    while True:
        Vn = np.empty(S)
        for s in range(S):
            Vn[s] = max(R[s, a] + gamma * P[a, s] @ V for a in range(A))
        if np.abs(Vn - V).max() <= eps:
            return Vn
        V = Vn


print("stok")
