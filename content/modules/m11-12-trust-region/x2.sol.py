import numpy as np


def surrogates(t_grid, theta0, d, A, eps=0.2, beta=1.0):
    th = theta0[None] + t_grid[:, None] * d[None]
    e = np.exp(th - th.max(1, keepdims=True))
    pi = e / e.sum(1, keepdims=True)
    e0 = np.exp(theta0 - theta0.max())
    p0 = e0 / e0.sum()
    r = pi / p0
    cpi = (p0 * r * A).sum(1)
    clip = (p0 * np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)).sum(1)
    kl = (p0 * (np.log(p0) - np.log(pi))).sum(1)
    return cpi, clip, cpi - beta * kl, kl


t = np.linspace(-3, 3, 7)
print(surrogates(t, np.zeros(3), np.array([1.0, -1.0, 0.0]), np.array([1.0, -0.5, 0.0])))
