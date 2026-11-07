import numpy as np


def xavier_std(fan_in, fan_out):
    ...


def kaiming_std(fan_in):
    ...


def activation_stds(X, depth, width, std, act, rng):
    ...


relu = lambda z: np.maximum(0, z)
X = np.random.default_rng(0).standard_normal((500, 256))
print(activation_stds(X, 10, 256, kaiming_std(256), relu, np.random.default_rng(1))[-1])
