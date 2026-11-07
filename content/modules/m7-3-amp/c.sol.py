import numpy as np


def train_fp16(X, y, lr, steps, master):
    X16, y16 = X.astype(np.float16), y.astype(np.float16)
    w32 = np.ones(X.shape[1], dtype=np.float32)
    w = w32.astype(np.float16)
    n = np.float16(X.shape[0])
    for _ in range(steps):
        r = X16 @ w - y16
        g = (np.float16(2) * (X16.T @ r) / n).astype(np.float16)
        if master:
            w32 -= np.float32(lr) * g.astype(np.float32)
            w = w32.astype(np.float16)
        else:
            w = (w - np.float16(lr) * g).astype(np.float16)
    return float(np.mean((X @ w.astype(np.float32) - y) ** 2))


def scaled_grad(g, scale):
    return np.float16(g), np.float32(np.float16(g * scale)) / np.float32(scale)


rng = np.random.default_rng(0)
X = rng.normal(size=(256, 10)).astype(np.float32)
w_true = (1 + 0.003 * np.arange(10)).astype(np.float32)
y = X @ w_true
print(train_fp16(X, y, 0.01, 300, master=False), train_fp16(X, y, 0.01, 300, master=True))
print(scaled_grad(1e-8, 65536.0))
