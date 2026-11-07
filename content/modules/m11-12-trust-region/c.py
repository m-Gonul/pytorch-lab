import numpy as np


def softmax(theta):
    ...


def kl(p, q):
    ...


def fisher(pi):
    ...


def fvp(pi, v):
    ...


def policy_grad(theta, r):
    ...


def cg(Avp, b, iters=20, tol=1e-12):
    ...


def natural_step(theta, r, delta, damping=1e-3):
    ...


th = np.zeros((1, 3)); r = np.array([[1.0, 0.0, 0.5]])
print(natural_step(th, r, 0.01))
