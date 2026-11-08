import numpy as np


def train_neuron(w0, b0, x, y, loss, lr, steps):
    w, b, out = w0, b0, []
    for _ in range(steps):
        a = 1 / (1 + np.exp(-(w * x + b)))
        out.append(a)
        d = (a - y) * a * (1 - a) if loss == "mse" else (a - y)
        w -= lr * d * x
        b -= lr * d
    return out
