import math
import random
N = 10


def chain_step(s, a):
    if a == 0:
        return 0, (0.05 if s == 0 else 0.0)
    if s == N - 1:
        return 0, 1.0
    return s + 1, 0.0


def run(strategy, Q0=0.0, n_steps=5000, gamma=0.95, alpha=0.2, seed=0):
    ...


print(run("eps"), run("greedy", Q0=2.0))
