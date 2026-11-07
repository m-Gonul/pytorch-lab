import random


def sample_index(probs, u):
    ...


def simulate_chain(T, s0, steps, seed=0):
    ...


def ar1(phi, sigma, n, seed=0):
    ...


def autocov(x, lag):
    ...


T = [[0.9, 0.5], [0.1, 0.5]]
s = simulate_chain(T, 0, 10000)
print(s.count(0) / len(s))
