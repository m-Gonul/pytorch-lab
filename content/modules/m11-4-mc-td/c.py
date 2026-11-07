import numpy as np


def batch_mc(episodes, n_states):
    ...


def batch_td(episodes, n_states, alpha=0.01, tol=1e-10):
    ...


def certainty_equivalence(episodes, n_states):
    ...


AB = [[(0, 0.0), (1, 0.0)]] + [[(1, 1.0)]] * 6 + [[(1, 0.0)]]
print(batch_mc(AB, 2), batch_td(AB, 2))
