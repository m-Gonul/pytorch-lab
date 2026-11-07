import numpy as np


def bias_vs_weight_grads(a_prev, delta):
    a, d = np.asarray(a_prev, float), np.asarray(delta, float)
    return d.copy(), np.outer(d, a)


def zigzag_fraction(n_steps, act, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, (n_steps, 2))
    a = 1 / (1 + np.exp(-z)) if act == "sigmoid" else np.tanh(z)
    delta = rng.normal(size=n_steps)
    g = delta[:, None] * a
    return float((np.sign(g[:, 0]) == np.sign(g[:, 1])).mean())


def derivative_max(act):
    z = np.linspace(-10, 10, 200001)
    if act == "sigmoid":
        s = 1 / (1 + np.exp(-z))
        return float((s * (1 - s)).max())
    return float((1 - np.tanh(z) ** 2).max())


print(derivative_max("sigmoid"), derivative_max("tanh"))
