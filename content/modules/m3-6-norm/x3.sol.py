import numpy as np


def loss_and_grad(w, X, v):
    N = len(X)
    z = X @ w
    mu, sd = z.mean(), z.std()
    zh = (z - mu) / sd
    L = float(v @ zh / N)
    dzh = v / N
    dz = (dzh - dzh.mean() - zh * (dzh @ zh) / N) / sd
    return L, X.T @ dz
