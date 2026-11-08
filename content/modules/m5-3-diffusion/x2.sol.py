import numpy as np


def posterior(x0, xt, t, betas):
    ab = np.cumprod(1 - betas)
    abp = ab[t - 1] if t >= 1 else 1.0
    a, b = 1 - betas[t], betas[t]
    mean = np.sqrt(abp) * b / (1 - ab[t]) * x0 + np.sqrt(a) * (1 - abp) / (1 - ab[t]) * xt
    return float(mean), float((1 - abp) / (1 - ab[t]) * b)


def mc_posterior(x0, xt, t, betas, rng, n, tol):
    ab = np.cumprod(1 - betas)
    abp = ab[t - 1] if t >= 1 else 1.0
    xp = np.sqrt(abp) * x0 + np.sqrt(1 - abp) * rng.normal(size=n)
    x = np.sqrt(1 - betas[t]) * xp + np.sqrt(betas[t]) * rng.normal(size=n)
    sel = xp[np.abs(x - xt) < tol]
    return float(sel.mean()), float(sel.var())
