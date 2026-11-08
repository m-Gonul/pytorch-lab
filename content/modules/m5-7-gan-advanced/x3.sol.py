import numpy as np


def fade_in(old_lowres, new_highres, alpha):
    up = old_lowres.repeat(2, 1).repeat(2, 2)
    return (1 - alpha) * up + alpha * new_highres


def minibatch_std(X, eps=1e-8):
    s = np.sqrt(X.var(0) + eps).mean()
    return np.concatenate([X, np.full((X.shape[0], 1) + X.shape[2:], s)], 1)


def pixel_norm(X, eps=1e-8):
    return X / np.sqrt((X ** 2).mean(1, keepdims=True) + eps)


def equalized_step(fan_in, lr, steps, seed):
    rng = np.random.default_rng(seed)
    c = np.sqrt(2 / fan_in)
    wa = rng.normal(0, c, fan_in)
    wb = rng.normal(0, 1, fan_in)
    a0, b0 = wa.copy(), c * wb
    for _ in range(steps):
        g = rng.normal(size=fan_in)
        wa = wa - lr * np.sign(g)
        wb = wb - lr * np.sign(c * g)
    return float(np.linalg.norm(wa - a0) / np.linalg.norm(a0)), float(np.linalg.norm(c * wb - b0) / np.linalg.norm(b0))
