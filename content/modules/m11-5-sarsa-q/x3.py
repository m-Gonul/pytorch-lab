import random
import numpy as np


def draw(rng):
    return min(rng.randint(1, 13), 10)


def mc_es(n, seed=0):
    ...


Q, pol = mc_es(20000)
print(pol[0, ::-1])
