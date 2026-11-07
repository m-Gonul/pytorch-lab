import numpy as np


def softmax(x, axis=-1):
    ...


def mha(x, Wq, Wk, Wv, Wo, h, causal=False):
    ...


rng = np.random.default_rng(0)
x = rng.normal(size=(2, 5, 8)); W = rng.normal(size=(4, 8, 8)) / 3
print(mha(x, *W, h=2, causal=True).shape)   # (2, 5, 8)
