import math


def bn_forward(X, gamma, beta, eps=1e-5):
    ...


def bn_backward(dout, cache):
    ...


def rms_norm(x, g, eps=1e-6):
    ...


out, cache = bn_forward([[1.0], [2.0], [3.0]], [2.0], [1.0], eps=0.0)
print(out)                                   # [[-1.449], [1.0], [3.449]]
print(bn_backward([[1.0], [0.0], [0.0]], cache))
