import math
import numpy as np

LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
BOARD = [1, 0, 0, -1, 1, 0, 0, -1, 0]  # X'in sırası; 8'e oynarsa (çapraz 0-4-8) kazanır
WIN = 8


def root_visits(n_sims, noise, rng):
    ...


def win_rate(noise, n_trials=200, n_sims=8, seed=0):
    ...


print(win_rate(False), win_rate(True))
