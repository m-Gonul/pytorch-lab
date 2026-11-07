import numpy as np


def gauss_reinforce(n_iter, batch=64, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    mu, ls = 0.0, 0.0
    mus, sigmas = [mu], [np.exp(ls)]
    for _ in range(n_iter):
        s = np.exp(ls)
        a = mu + s * rng.normal(0, 1, batch)
        r = -(a - 3) ** 2 + 0.1 * rng.normal(0, 1, batch)
        adv = r - r.mean()
        mu += lr * np.mean(adv * (a - mu) / s ** 2) * s ** 2
        ls += lr * np.mean(adv * ((a - mu) ** 2 / s ** 2 - 1)) * 0.2
        mus.append(mu)
        sigmas.append(np.exp(ls))
    return mus, sigmas


mus, sigmas = gauss_reinforce(200)
print(mus[-1], sigmas[-1])
