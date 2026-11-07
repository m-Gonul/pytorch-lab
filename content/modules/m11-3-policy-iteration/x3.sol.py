import numpy as np


def gambler_vi(ph=0.4, eps=1e-12):
    V = np.zeros(101); V[100] = 1.0
    s = np.arange(1, 100)
    a = np.arange(1, 51)
    ok = a[None, :] <= np.minimum(s, 100 - s)[:, None]
    up = np.minimum(s[:, None] + a[None, :], 100)
    dn = np.maximum(s[:, None] - a[None, :], 0)
    while True:
        Q = np.where(ok, ph * V[up] + (1 - ph) * V[dn], -np.inf)
        Vn = V.copy(); Vn[1:100] = Q.max(1)
        if np.abs(Vn - V).max() <= eps:
            V = Vn
            break
        V = Vn
    Q = np.where(ok, ph * V[up] + (1 - ph) * V[dn], -np.inf)
    pi = np.zeros(101, int)
    pi[1:100] = a[np.argmax(np.round(Q, 9), 1)]
    return V, pi


V, pi = gambler_vi()
print(pi[1:100:7])
