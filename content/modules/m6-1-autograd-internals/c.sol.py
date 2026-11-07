import numpy as np


class Var:
    def __init__(self, data, parents=(), back=None):
        self.data = np.asarray(data, dtype=float)
        self.version = 0
        self.grad = None
        self.parents = parents
        self._back = back      # g -> girdilerin gradyanları (parents sırasıyla)


# ---- senin kodun ----

def _check(saved):
    for v, ver in saved:
        if v.version != ver:
            raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")


def mul(a, b):
    saved = [(a, a.version), (b, b.version)]

    def back(g):
        _check(saved)
        return g * b.data, g * a.data
    return Var(a.data * b.data, (a, b), back)


def exp(a):
    out = Var(np.exp(a.data), (a,))
    saved = [(out, out.version)]

    def back(g):
        _check(saved)
        return (g * out.data,)
    out._back = back
    return out


def sum_(a):
    return Var(a.data.sum(), (a,), lambda g: (g * np.ones_like(a.data),))


def add_(a, c):
    a.data += c
    a.version += 1
    return a


def backward(out):
    order, seen = [], set()

    def dfs(v):
        if id(v) in seen:
            return
        seen.add(id(v))
        for p in v.parents:
            dfs(p)
        order.append(v)
    dfs(out)
    grads = {id(out): np.ones_like(out.data)}
    for v in reversed(order):
        g = grads.pop(id(v), None)
        if g is None:
            continue
        if not v.parents:
            v.grad = g if v.grad is None else v.grad + g
            continue
        for p, gp in zip(v.parents, v._back(g)):
            grads[id(p)] = grads.get(id(p), 0) + gp


x, w = Var([0.5, -1.0]), Var([2.0, 3.0])
s = sum_(exp(mul(x, w)))
backward(s)
print(x.grad, w.grad)
