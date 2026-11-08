import numpy as np


def sign_agreement(A, delta):
    G = delta[:, :, None] * A[:, None, :]
    pos = (G > 0).all(-1)
    neg = (G < 0).all(-1)
    return float((pos | neg).mean())


def saturation(Z, act):
    if act == "sigmoid":
        s = 1 / (1 + np.exp(-Z))
        d, dmax = s * (1 - s), 0.25
    else:
        d, dmax = 1 - np.tanh(Z) ** 2, 1.0
    return float((d < 0.1 * dmax).mean())
