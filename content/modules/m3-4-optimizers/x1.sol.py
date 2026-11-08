import numpy as np


def ema(g, beta, bias_correct=False):
    m, out = 0.0, []
    for t, v in enumerate(g, 1):
        m = beta * m + (1 - beta) * v
        out.append(m / (1 - beta ** t) if bias_correct else m)
    return np.array(out)


def heavy_ball(A, x0, lr, beta, steps):
    x, v, L = x0.astype(float), np.zeros_like(x0, float), []
    for _ in range(steps):
        v = beta * v - lr * (A @ x)
        x = x + v
        L.append(0.5 * x @ A @ x)
    return np.array(L)


def nesterov(A, x0, lr, beta, steps):
    x, v, L = x0.astype(float), np.zeros_like(x0, float), []
    for _ in range(steps):
        v = beta * v - lr * (A @ (x + beta * v))
        x = x + v
        L.append(0.5 * x @ A @ x)
    return np.array(L)
