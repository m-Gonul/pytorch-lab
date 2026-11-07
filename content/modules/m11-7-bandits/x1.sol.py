import numpy as np


def coverage(p, n, c, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.binomial(n, p, n_trials) / n
    return float((np.abs(means - p) >= c - 1e-12).mean()), 2 * np.exp(-2 * n * c * c)


print(coverage(0.5, 100, 0.1, 100000))
