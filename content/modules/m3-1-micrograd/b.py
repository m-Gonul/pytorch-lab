import math


class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_prev)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        ...

    def __pow__(self, k):
        ...

    def tanh(self):
        ...

    def backward(self):
        ...

    # kolaylıklar
    def __radd__(self, other): ...
    def __rmul__(self, other): ...
    def __neg__(self): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __truediv__(self, other): ...


a, b, c, f = Value(2), Value(-3), Value(10), Value(-2)
L = (a * b + c) * f
L.backward()
print(L, a.grad, b.grad, c.grad, f.grad)   # -8, 6, -4, -2, 4
