import math
import random


def alpha_bars(betas):
    ...


def q_sample(x0, t, eps, abar):
    ...


def oracle_eps(xt, t, abar, mu0, s0):
    ...


def ddpm_step(xt, t, eps_pred, betas, abar, z):
    ...


def sample(eps_fn, betas, n, rng):
    ...


T = 100
betas = [1e-4 + (0.1 - 1e-4) * i / (T - 1) for i in range(T)]
ab = alpha_bars(betas)
xs = sample(lambda x, t: oracle_eps(x, t, ab, 3.0, 0.5), betas, 2000, random.Random(0))
m = sum(xs) / len(xs)
print(m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5)   # ≈ 3, 0.5
