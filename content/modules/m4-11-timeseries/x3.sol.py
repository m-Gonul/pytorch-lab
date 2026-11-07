import numpy as np


def nlms(y, p, mu=0.1, eps=1e-6):
    y = np.asarray(y, float)
    w = np.zeros(p)
    pred = np.zeros(len(y))
    W = np.zeros((len(y), p))
    for t in range(p, len(y)):
        u = y[t - p:t][::-1]
        pred[t] = w @ u
        e = y[t] - pred[t]
        w = w + mu * e * u / (eps + u @ u)
        W[t] = w
    return pred, W
