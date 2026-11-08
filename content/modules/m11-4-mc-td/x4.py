import numpy as np


def chain_step(s, a, N):
    """Zincir: a = 0 sola, 1 sağa. Solda (s = 0) sola gitmek 0.05 ödül, sağ uçta (s = N−1) sağa gitmek 1 ödül."""
    if a == 0:
        return max(s - 1, 0), (0.05 if s == 0 else 0.0)
    return min(s + 1, N - 1), (1.0 if s == N - 1 else 0.0)


def select(q, method, t, rng, eps=0.1, tau=0.1):
    ...


def q_learning(N, episodes, method, q0, seed, alpha=0.2, gamma=0.95):
    ...
