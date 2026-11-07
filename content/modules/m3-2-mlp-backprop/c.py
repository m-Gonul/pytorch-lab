import numpy as np


def affine_forward(x, w, b):
    ...


def affine_backward(dout, cache):
    ...


def relu_forward(x):
    ...


def relu_backward(dout, cache):
    ...


def softmax_loss(scores, y):
    ...


def two_layer_loss(X, y, params, reg=0.0):
    ...


def train(X, y, H, C, lr=1.0, steps=5000, reg=1e-3, seed=0):
    ...


def spiral(n=100, K=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(K):
        r = np.linspace(0.0, 1, n)
        t = np.linspace(k * 4, (k + 1) * 4, n) + rng.normal(size=n) * 0.2
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y += [k] * n
    return np.concatenate(X), np.array(y)


X, y = spiral()
p = train(X, y, 100, 3)
h = np.maximum(0, X @ p["W1"] + p["b1"])
print("doğruluk:", ((h @ p["W2"] + p["b2"]).argmax(1) == y).mean())
