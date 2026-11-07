import numpy as np


def q_sample(x0, t, eps, abar):
    a = abar[t][:, None]
    return np.sqrt(a) * x0 + np.sqrt(1 - a) * eps


def ddpm_sample(eps_fn, betas, n, d, rng):
    abar = np.cumprod(1 - betas)
    x = rng.normal(size=(n, d))
    for t in range(len(betas) - 1, -1, -1):
        e = eps_fn(x, t)
        x = (x - betas[t] / np.sqrt(1 - abar[t]) * e) / np.sqrt(1 - betas[t])
        if t > 0:
            x = x + np.sqrt((1 - abar[t - 1]) / (1 - abar[t]) * betas[t]) * rng.normal(size=(n, d))
    return x


def ddim_sample(eps_fn, betas, n, d, rng, steps):
    abar = np.cumprod(1 - betas)
    ts = np.linspace(len(betas) - 1, 0, steps).round().astype(int)
    prev = np.append(abar[ts[1:]], 1.0)
    x = rng.normal(size=(n, d))
    for t, ap in zip(ts, prev):
        e = eps_fn(x, t)
        x0 = (x - np.sqrt(1 - abar[t]) * e) / np.sqrt(abar[t])
        x = np.sqrt(ap) * x0 + np.sqrt(1 - ap) * e
    return x


mu, s = np.array([2.0, -1.0]), np.array([0.5, 1.5])
betas = np.linspace(1e-4, 0.05, 300)
abar = np.cumprod(1 - betas)
oracle = lambda x, t: np.sqrt(1 - abar[t]) * (x - np.sqrt(abar[t]) * mu) / (abar[t] * s ** 2 + 1 - abar[t])
x = ddim_sample(oracle, betas, 4000, 2, np.random.default_rng(0), 50)
print(x.mean(0), x.std(0))
