import numpy as np


def random_phase(n_paths, n, omega, seed=0):
    th = np.random.default_rng(seed).uniform(0, 2 * np.pi, (n_paths, 1))
    return np.cos(omega * np.arange(n)[None] + th)


def random_amplitude(n_paths, n, omega, seed=0):
    A = np.random.default_rng(seed).normal(size=(n_paths, 1))
    return A * np.cos(omega * np.arange(n)[None])


def random_constant(n_paths, n, seed=0):
    return np.repeat(np.random.default_rng(seed).normal(size=(n_paths, 1)), n, 1)


def time_vs_ensemble(X):
    return float(X.mean(1).std()), float(X.mean(0).std())


print(random_phase(3, 5, 0.3).shape)
