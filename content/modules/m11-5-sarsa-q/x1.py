import random

WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


def windy_step(r, c, a):
    ...


def sarsa_windy(n_steps, seed=0):
    ...


def greedy_len(Q):
    ...


Q, ep = sarsa_windy(8000)
print(ep[-1], greedy_len(Q))
