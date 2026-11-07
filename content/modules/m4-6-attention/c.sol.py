import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def mha(x, Wq, Wk, Wv, Wo, h, causal=False):
    B, T, d = x.shape
    dk = d // h
    split = lambda t: t.reshape(B, T, h, dk).transpose(0, 2, 1, 3)
    q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)
    S = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)
    if causal:
        S = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, S)
    o = softmax(S) @ v
    return o.transpose(0, 2, 1, 3).reshape(B, T, d) @ Wo


rng = np.random.default_rng(0)
x = rng.normal(size=(2, 5, 8)); W = rng.normal(size=(4, 8, 8)) / 3
print(mha(x, *W, h=2, causal=True).shape)   # (2, 5, 8)
