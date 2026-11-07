import numpy as np


def train_fp16(X, y, lr, steps, master):
    ...


def scaled_grad(g, scale):
    ...


rng = np.random.default_rng(0)
X = rng.normal(size=(256, 10)).astype(np.float32)
w_true = (1 + 0.003 * np.arange(10)).astype(np.float32)
y = X @ w_true
print(train_fp16(X, y, 0.01, 300, master=False), train_fp16(X, y, 0.01, 300, master=True))
print(scaled_grad(1e-8, 65536.0))
