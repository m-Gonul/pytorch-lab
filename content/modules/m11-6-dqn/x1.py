import math
import random


def mc_step(p, v, a):
    ...


def tiles(p, v, a, n_tilings=8, n=8):
    ...


def sarsa_mc(n_eps, alpha=0.5, seed=0, n_tilings=8, max_steps=5000):
    ...


w, lens = sarsa_mc(20)
print(lens)
