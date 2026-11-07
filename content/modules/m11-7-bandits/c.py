import numpy as np


def testbed(n_runs, n_steps, eps, Q0=0.0, alpha=None, seed=0):
    ...


def gradient_bandit(n_runs, n_steps, alpha, baseline=True, seed=0, offset=4.0):
    ...


r, opt = testbed(200, 500, 0.1)
print(r[-50:].mean(), opt[-50:].mean())
