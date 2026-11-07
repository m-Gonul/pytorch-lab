import random
import numpy as np


def draw(rng):
    return min(rng.randint(1, 13), 10)


def play(rng):
    ...


def mc_blackjack(n, seed=0):
    ...


print(mc_blackjack(1000)[0, 9, 0])
