import numpy as np


def clip_value(grads, c):
    return [np.clip(g, -c, c) for g in grads]


def clip_global_norm(grads, c):
    n = float(np.sqrt(sum(np.sum(g * g) for g in grads)))
    s = min(1.0, c / max(n, 1e-12))
    return [g * s for g in grads], n
