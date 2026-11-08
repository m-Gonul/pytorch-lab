import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def backprop(Ws, bs, x, y):
    a, As, Zs = x, [x], []
    for W, b in zip(Ws, bs):
        z = W @ a + b
        a = sigmoid(z)
        Zs.append(z)
        As.append(a)
    sp = [sigmoid(z) * (1 - sigmoid(z)) for z in Zs]
    deltas = [None] * len(Ws)
    deltas[-1] = (As[-1] - y) * sp[-1]
    for l in range(len(Ws) - 2, -1, -1):
        deltas[l] = (Ws[l + 1].T @ deltas[l + 1]) * sp[l]
    dWs = [d @ A.T for d, A in zip(deltas, As[:-1])]
    return deltas, dWs, [d.copy() for d in deltas]
