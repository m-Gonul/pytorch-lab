import math
import random


def alpha_bars(betas):
    out, p = [], 1.0
    for b in betas:
        p *= 1 - b
        out.append(p)
    return out


def q_sample(x0, t, eps, abar):
    return math.sqrt(abar[t]) * x0 + math.sqrt(1 - abar[t]) * eps


def oracle_eps(xt, t, abar, mu0, s0):
    a = abar[t]
    return math.sqrt(1 - a) * (xt - math.sqrt(a) * mu0) / (a * s0 ** 2 + 1 - a)


def ddpm_step(xt, t, eps_pred, betas, abar, z):
    mu = (xt - betas[t] / math.sqrt(1 - abar[t]) * eps_pred) / math.sqrt(1 - betas[t])
    if t == 0:
        return mu
    var = (1 - abar[t - 1]) / (1 - abar[t]) * betas[t]
    return mu + math.sqrt(var) * z


def sample(eps_fn, betas, n, rng):
    ab = alpha_bars(betas)
    out = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        for t in reversed(range(len(betas))):
            x = ddpm_step(x, t, eps_fn(x, t), betas, ab, rng.gauss(0, 1))
        out.append(x)
    return out


T = 100
betas = [1e-4 + (0.1 - 1e-4) * i / (T - 1) for i in range(T)]
ab = alpha_bars(betas)
xs = sample(lambda x, t: oracle_eps(x, t, ab, 3.0, 0.5), betas, 2000, random.Random(0))
m = sum(xs) / len(xs)
print(m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5)   # ≈ 3, 0.5
