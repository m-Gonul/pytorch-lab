import numpy as np

LAM, ALPHA = 1.0507009873554805, 1.6732632423543772


def alpha_dropout(x, p, rng):
    q = 1 - p
    ap = -LAM * ALPHA
    a = (q + ap ** 2 * q * (1 - q)) ** -0.5
    b = -a * (1 - q) * ap
    m = rng.random(x.shape) < q
    return a * np.where(m, x, ap) + b


def std_dropout(x, p, rng):
    return x * (rng.random(x.shape) >= p) / (1 - p)
