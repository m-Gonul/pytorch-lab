import math
import random

H, W = 6, 9
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
START, GOAL = (5, 3), (0, 8)
WALL_A = {(3, c) for c in range(1, 9)}
WALL_B = {(3, c) for c in range(1, 8)}


def run(plus, n_steps=6000, change=3000, kappa=1e-3, seed=0, n_plan=50):
    ...


print(run(True, 2000)[-1])
