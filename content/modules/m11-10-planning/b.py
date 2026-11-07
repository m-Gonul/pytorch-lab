import random

H, W = 6, 9
WALLS = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}
START, GOAL = (2, 0), (0, 8)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def maze_step(s, a):
    ...


def dyna_q(n_eps, n_plan, seed=0, alpha=0.1, gamma=0.95, eps=0.1):
    ...


print(dyna_q(20, 5)[1])
