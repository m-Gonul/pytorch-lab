import numpy as np


def q_sample(x0, t, eps, abar):
    ...


def ddpm_sample(eps_fn, betas, n, d, rng):
    ...


def ddim_sample(eps_fn, betas, n, d, rng, steps):
    ...


mu, s = np.array([2.0, -1.0]), np.array([0.5, 1.5])
betas = np.linspace(1e-4, 0.05, 300)
abar = np.cumprod(1 - betas)
oracle = lambda x, t: np.sqrt(1 - abar[t]) * (x - np.sqrt(abar[t]) * mu) / (abar[t] * s ** 2 + 1 - abar[t])
x = ddim_sample(oracle, betas, 4000, 2, np.random.default_rng(0), 50)
print(x.mean(0), x.std(0))
