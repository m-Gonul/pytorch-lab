import math
import random


def corridor_step(s, right):
    ...


def corridor_value(p):
    ...


def reinforce(n_eps, alpha, seed=0, baseline=False, alpha_w=0.0, max_steps=1000):
    ...


theta, ps = reinforce(200, 2 ** -10, baseline=True, alpha_w=2 ** -6)
print(ps[-1], corridor_value(ps[-1]))
