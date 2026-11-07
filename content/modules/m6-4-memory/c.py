import numpy as np


def grads_full(x, Ws, gout):
    ...


def grads_ckpt(x, Ws, gout, k):
    ...


rng = np.random.default_rng(0)
Ws = [rng.normal(size=(8, 8)) / 3 for _ in range(16)]
x, gout = rng.normal(size=(4, 8)), rng.normal(size=(4, 8))
g1 = grads_full(x, Ws, gout)
g2, peak = grads_ckpt(x, Ws, gout, 4)
print(max(np.abs(a - b).max() for a, b in zip(g1, g2)), peak)   # ~0, 7
