import numpy as np
import torch
from scipy.stats import poisson


def location(lam_req, lam_ret, N=20):
    ...


def jack_pi(gamma=0.9):
    ...


V, pi, it = jack_pi()
print(it, pi[20, 0], round(float(V[0, 0]), 1))
