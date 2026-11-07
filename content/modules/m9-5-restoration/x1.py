import numpy as np


def make_noise(kind, shape, rng, **p):
    ...


def estimate(kind, samples):
    ...


def add_salt_pepper(img, ps, pp, rng):
    ...


rng = np.random.default_rng(0)
z = make_noise("rayleigh", (100000,), rng, a=10, b=400)
print(estimate("rayleigh", z))
