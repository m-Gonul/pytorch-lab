import numpy as np


def elo_diff_mle(wins, losses):
    return 400 * np.log10(wins / losses)


def elo_posterior(wins, losses, n_samples=200000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(1 + wins, 1 + losses, n_samples)
    d = 400 * np.log10(p / (1 - p))
    return float(np.median(d)), float(np.percentile(d, 5)), float(np.percentile(d, 95))


print(elo_diff_mle(4, 1), elo_posterior(4, 1))
