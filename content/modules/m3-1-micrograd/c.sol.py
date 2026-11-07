import numpy as np


def neuron_fb(X, w, b):
    """o = tanh(Xw + b), L = Σo. → (o, L, dL/dw, dL/db). Döngü yok."""
    o = np.tanh(X @ w + b)
    L = o.sum()
    g = 1 - o ** 2              # ∂L/∂n, (n,)
    return o, L, X.T @ g, g.sum()


X = np.array([[2.0, 0.0], [1.0, -1.0], [0.0, 0.5]])
w = np.array([-3.0, 1.0])
print(neuron_fb(X, w, 6.8813735870195432))
