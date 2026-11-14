import random
import numpy as np


def draw(rng):
    return min(rng.randint(1, 13), 10)


def play(rng, natural_rule=True):
    ...


def mc_blackjack(n, seed=0, return_counts=False, natural_rule=True):
    ...


print(mc_blackjack(1000)[0, 9, 0])
