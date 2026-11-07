import numpy as np


def lost_update_prob(L, T, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, T, (n_trials, 2))
    return float((np.abs(s[:, 0] - s[:, 1]) < L).mean())


def expected_lost(k, L, T, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    s = np.sort(rng.uniform(0, T, (n_trials, k)), 1)
    comps = 1 + (np.diff(s, axis=1) >= L).sum(1)
    return float((k - comps).mean())


print(lost_update_prob(0.1, 1.0, 100000), 1 - 0.9 ** 2)
