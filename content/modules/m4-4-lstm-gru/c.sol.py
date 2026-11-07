import numpy as np


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    H = prev_h.shape[1]
    a = x @ Wx + prev_h @ Wh + b
    i, f, g, o = sigmoid(a[:, :H]), sigmoid(a[:, H:2 * H]), np.tanh(a[:, 2 * H:3 * H]), sigmoid(a[:, 3 * H:])
    next_c = f * prev_c + i * g
    tc = np.tanh(next_c)
    return o * tc, next_c, (x, prev_h, prev_c, Wx, Wh, i, f, g, o, tc)


def lstm_step_backward(dnext_h, dnext_c, cache):
    """→ (dx, dprev_h, dprev_c, dWx, dWh, db)"""
    x, prev_h, prev_c, Wx, Wh, i, f, g, o, tc = cache
    dc = dnext_c + dnext_h * o * (1 - tc ** 2)
    da = np.concatenate([dc * g * i * (1 - i),
                         dc * prev_c * f * (1 - f),
                         dc * i * (1 - g ** 2),
                         dnext_h * tc * o * (1 - o)], axis=1)
    return da @ Wx.T, da @ Wh.T, dc * f, x.T @ da, prev_h.T @ da, da.sum(0)
