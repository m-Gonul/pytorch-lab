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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "yalnız sayı üsler"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += out.grad * k * self.data ** (k - 1)
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1 - t * t)
        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # kolaylıklar
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * (other if isinstance(other, Value) else Value(other)) ** -1


a, b, c, f = Value(2), Value(-3), Value(10), Value(-2)
L = (a * b + c) * f
L.backward()
print(L, a.grad, b.grad, c.grad, f.grad)   # -8, 6, -4, -2, 4
