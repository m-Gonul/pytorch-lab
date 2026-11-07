import torch

A_, B_, Q_, R_, GAM = 1.1, 0.5, 1.0, 0.5, 0.95


def riccati(iters=2000):
    P = 0.0
    for _ in range(iters):
        P = Q_ + GAM * A_ * A_ * P - (GAM * A_ * B_ * P) ** 2 / (R_ + GAM * B_ * B_ * P)
    return P, -GAM * A_ * B_ * P / (R_ + GAM * B_ * B_ * P)


def collect(k, n_envs, T, sigma, g):
    ...


def lstdq(S, U, R, S2, k):
    ...


def dpg(n_iter=40, k0=-0.5, alpha=None, n_envs=256, T=20, sigma=0.5, seed=0):
    ...


print(riccati()[1], dpg()[0])
