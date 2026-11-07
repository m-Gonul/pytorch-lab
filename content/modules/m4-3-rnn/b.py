import math


def rnn_forward(xs, h0, Wx, Wh, b):
    ...


def rnn_backward(xs, hs, h0, Wx, Wh, dhs):
    """→ (dWx, dWh, db, dh0)"""
    ...


hs = rnn_forward([[1.0], [0.0], [0.0]], [0.0], [[1.0]], [[0.5]], [0.0])
print(hs)                                                     # [[0.7616], [0.3634], [0.1797]]
print(rnn_backward([[1.0], [0.0], [0.0]], hs, [0.0], [[1.0]], [[0.5]], [[0.0], [0.0], [1.0]]))
