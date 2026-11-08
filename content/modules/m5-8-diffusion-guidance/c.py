import numpy as np

ANG = np.arange(8) * np.pi / 4
MU = 4 * np.stack([np.cos(ANG), np.sin(ANG)], 1)
LAB = np.arange(8) % 4
SIG = 0.5


def mixture_eps(x, abar, w):
    ...


def class_logprob_grad(x, abar, y):
    ...


def ddim(xT, abars, eps_fn):
    ...
