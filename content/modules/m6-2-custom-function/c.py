import numpy as np


def ce_forward(logits, y):
    ...


def ce_backward(cache):
    ...


def dir_gradcheck(f, grad, x, rng, k=10, eps=1e-6):
    ...


rng = np.random.default_rng(0)
Z, y = rng.normal(size=(5, 4)), rng.integers(0, 4, 5)
loss, cache = ce_forward(Z, y)
print(loss, ce_backward(cache).sum(1))     # satır toplamları 0
