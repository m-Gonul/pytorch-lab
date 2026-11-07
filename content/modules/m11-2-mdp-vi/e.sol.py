import numpy as np
import torch
from scipy.stats import poisson


def inventory_mdp(C=50, max_order=20, lam=8.0, price=10.0, cost=4.0, hold=0.5, short=2.0):
    s = np.arange(C + 1)[:, None]
    a = np.arange(max_order + 1)[None, :]
    y = np.minimum(s + a, C)
    Dmax = C + max_order
    d = np.arange(Dmax + 1)
    pd = poisson.pmf(d, lam)
    pd[-1] += poisson.sf(Dmax, lam)
    left = np.maximum(y[..., None] - d, 0)
    sold = np.minimum(y[..., None], d)
    unmet = np.maximum(d - y[..., None], 0)
    R = (pd * (price * sold - hold * left - short * unmet)).sum(-1) - cost * a
    P = np.zeros((max_order + 1, C + 1, C + 1))
    np.add.at(P, (np.broadcast_to(a[..., None], left.shape), np.broadcast_to(s[..., None], left.shape), left), np.broadcast_to(pd, left.shape))
    return P, R


def vi_vectorized(P, R, gamma, eps=1e-8):
    Pt, Rt = torch.tensor(P), torch.tensor(R)
    V = torch.zeros(P.shape[1], dtype=torch.float64)
    while True:
        Vn = (Rt + gamma * torch.einsum("ast,t->sa", Pt, V)).max(1).values
        if (Vn - V).abs().max() <= eps:
            return Vn.numpy(), (Rt + gamma * torch.einsum("ast,t->sa", Pt, Vn)).argmax(1).numpy()
        V = Vn


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
