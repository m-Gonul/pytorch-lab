import numpy as np


def neuron_fb(X, w, b):
    """o = tanh(Xw + b), L = Σo. → (o, L, dL/dw, dL/db). Döngü yok."""
    ...


X = np.array([[2.0, 0.0], [1.0, -1.0], [0.0, 0.5]])
w = np.array([-3.0, 1.0])
print(neuron_fb(X, w, 6.8813735870195432))
