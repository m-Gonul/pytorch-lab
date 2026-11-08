import numpy as np

ANG = np.arange(8) * np.pi / 4
MU = 4 * np.stack([np.cos(ANG), np.sin(ANG)], 1)
LAB = np.arange(8) % 4
SIG = 0.5


def _resp(x, abar, w):
    v = abar * SIG ** 2 + 1 - abar
    m = np.sqrt(abar) * MU
    lw = np.log(np.clip(w, 1e-300, None))[None] - ((x[:, None] - m[None]) ** 2).sum(-1) / (2 * v)
    r = np.exp(lw - lw.max(1, keepdims=True))
    return r / r.sum(1, keepdims=True), m, v


def mixture_eps(x, abar, w):
    r, m, v = _resp(x, abar, w)
    return np.sqrt(1 - abar) * (x - r @ m) / v


def class_logprob_grad(x, abar, y):
    wu = np.full(8, 1 / 8)
    wc = (LAB == y) / (LAB == y).sum()
    return -(mixture_eps(x, abar, wc) - mixture_eps(x, abar, wu)) / np.sqrt(1 - abar)


def ddim(xT, abars, eps_fn):
    x = xT
    for i in range(len(abars) - 1, 0, -1):
        e = eps_fn(x, abars[i])
        x0 = (x - np.sqrt(1 - abars[i]) * e) / np.sqrt(abars[i])
        x = np.sqrt(abars[i - 1]) * x0 + np.sqrt(1 - abars[i - 1]) * e
    e = eps_fn(x, abars[0])
    return (x - np.sqrt(1 - abars[0]) * e) / np.sqrt(abars[0])
