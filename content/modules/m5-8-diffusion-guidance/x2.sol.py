import numpy as np

ANG = np.arange(8) * np.pi / 4
MU = 4 * np.stack([np.cos(ANG), np.sin(ANG)], 1)
LAB = np.arange(8) % 4
SIG = 0.5


def mixture_eps(x, abar, w):
    v = abar * SIG ** 2 + 1 - abar
    m = np.sqrt(abar) * MU
    lw = np.log(np.clip(w, 1e-300, None))[None] - ((x[:, None] - m[None]) ** 2).sum(-1) / (2 * v)
    r = np.exp(lw - lw.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    return np.sqrt(1 - abar) * (x - r @ m) / v


def ddim(xT, abars, eps_fn):
    x = xT
    for i in range(len(abars) - 1, 0, -1):
        e = eps_fn(x, abars[i])
        x0 = (x - np.sqrt(1 - abars[i]) * e) / np.sqrt(abars[i])
        x = np.sqrt(abars[i - 1]) * x0 + np.sqrt(1 - abars[i - 1]) * e
    e = eps_fn(x, abars[0])
    return (x - np.sqrt(1 - abars[0]) * e) / np.sqrt(abars[0])


def ddim_invert(x0, abars, eps_fn):
    x = np.sqrt(abars[0]) * x0 + np.sqrt(1 - abars[0]) * eps_fn(np.sqrt(abars[0]) * x0, abars[0])
    for i in range(len(abars) - 1):
        e = eps_fn(x, abars[i])
        xh = (x - np.sqrt(1 - abars[i]) * e) / np.sqrt(abars[i])
        x = np.sqrt(abars[i + 1]) * xh + np.sqrt(1 - abars[i + 1]) * e
    return x
