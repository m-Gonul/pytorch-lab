import numpy as np


def var_recursion(v0, betas, scaled=True):
    v, out = v0, []
    for b in betas:
        v = (1 - b) * v + b if scaled else v + b
        out.append(v)
    return np.array(out)


def forward_chain(x0, betas, rng, n):
    x = np.full(n, float(x0))
    for b in betas:
        x = np.sqrt(1 - b) * x + np.sqrt(b) * rng.normal(size=n)
    return x


def snr(abar):
    return np.asarray(abar) / (1 - np.asarray(abar))
