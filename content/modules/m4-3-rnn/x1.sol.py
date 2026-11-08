import numpy as np


def ln(a, gamma, beta, eps=1e-5):
    m = a.mean(-1, keepdims=True)
    v = a.var(-1, keepdims=True)
    return gamma * (a - m) / np.sqrt(v + eps) + beta


def rnn_run(X, Wx, Wh, b, ln_params=None):
    T, B, _ = X.shape
    h = np.zeros((B, Wh.shape[0]))
    out = []
    for t in range(T):
        a = X[t] @ Wx + h @ Wh + b
        if ln_params is not None:
            a = ln(a, *ln_params)
        h = np.tanh(a)
        out.append(h)
    return np.stack(out)
