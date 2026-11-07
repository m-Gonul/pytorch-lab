import numpy as np


def rnn_forward(x, h0, Wx, Wh, b):
    N, T, D = x.shape
    h = np.empty((N, T, Wh.shape[0]))
    prev = h0
    for t in range(T):
        prev = np.tanh(x[:, t] @ Wx + prev @ Wh + b)
        h[:, t] = prev
    return h, (x, h0, Wx, Wh, h)


def rnn_backward(dh, cache):
    """→ (dx, dh0, dWx, dWh, db)"""
    x, h0, Wx, Wh, h = cache
    N, T, D = x.shape
    dx = np.zeros_like(x)
    dWx, dWh, db = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros(Wh.shape[0])
    carry = np.zeros_like(h0)
    for t in reversed(range(T)):
        da = (dh[:, t] + carry) * (1 - h[:, t] ** 2)
        prev = h[:, t - 1] if t > 0 else h0
        dx[:, t] = da @ Wx.T
        dWx += x[:, t].T @ da
        dWh += prev.T @ da
        db += da.sum(0)
        carry = da @ Wh.T
    return dx, carry, dWx, dWh, db
