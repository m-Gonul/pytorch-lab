import numpy as np

MU = np.array([[-0.6, -2.0], [0.6, 2.0]])
SIG = 0.3
BETAS = np.linspace(1e-4, 0.05, 200)
AB = np.cumprod(1 - BETAS)


def eps(x, ab):
    v = ab * SIG ** 2 + 1 - ab
    m = np.sqrt(ab) * MU
    lw = -((x[:, None] - m[None]) ** 2).sum(-1) / (2 * v)
    r = np.exp(lw - lw.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    return np.sqrt(1 - ab) * (x - r @ m) / v


def repaint(xk, n, rng, U):
    ...
