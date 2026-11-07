import numpy as np


def affine_forward(x, w, b):
    return x @ w + b, (x, w)


def affine_backward(dout, cache):
    x, w = cache
    return dout @ w.T, x.T @ dout, dout.sum(0)


def relu_forward(x):
    return np.maximum(0, x), x


def relu_backward(dout, cache):
    return dout * (cache > 0)


def softmax_loss(scores, y):
    n = scores.shape[0]
    z = scores - scores.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    loss = -logp[np.arange(n), y].mean()
    d = np.exp(logp)
    d[np.arange(n), y] -= 1
    return loss, d / n


def two_layer_loss(X, y, params, reg=0.0):
    h, c1 = affine_forward(X, params["W1"], params["b1"])
    a, c2 = relu_forward(h)
    s, c3 = affine_forward(a, params["W2"], params["b2"])
    loss, ds = softmax_loss(s, y)
    loss += reg * ((params["W1"] ** 2).sum() + (params["W2"] ** 2).sum())
    da, dW2, db2 = affine_backward(ds, c3)
    dh = relu_backward(da, c2)
    _, dW1, db1 = affine_backward(dh, c1)
    return loss, {"W1": dW1 + 2 * reg * params["W1"], "b1": db1, "W2": dW2 + 2 * reg * params["W2"], "b2": db2}


def train(X, y, H, C, lr=1.0, steps=5000, reg=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    p = {"W1": 0.01 * rng.standard_normal((D, H)), "b1": np.zeros(H), "W2": 0.01 * rng.standard_normal((H, C)), "b2": np.zeros(C)}
    for _ in range(steps):
        _, g = two_layer_loss(X, y, p, reg)
        for k in p:
            p[k] -= lr * g[k]
    return p


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
