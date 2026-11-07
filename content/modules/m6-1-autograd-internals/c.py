import numpy as np


class Var:
    def __init__(self, data, parents=(), back=None):
        self.data = np.asarray(data, dtype=float)
        self.version = 0
        self.grad = None
        self.parents = parents
        self._back = back      # g -> girdilerin gradyanları (parents sırasıyla)


# ---- senin kodun ----

def mul(a, b):
    ...


def exp(a):
    ...


def sum_(a):
    ...


def add_(a, c):
    ...


def backward(out):
    ...


x, w = Var([0.5, -1.0]), Var([2.0, 3.0])
s = sum_(exp(mul(x, w)))
backward(s)
print(x.grad, w.grad)
