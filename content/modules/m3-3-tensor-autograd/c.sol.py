import numpy as np


def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    return grad.sum(axis=axes, keepdims=True) if axes else grad


class Tensor:
    def __init__(self, data, _prev=(), _op=""):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_prev)
        self._op = _op
        self._backward = lambda: None

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def __add__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Tensor(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += out.grad * k * self.data ** (k - 1)
        out._backward = _backward
        return out

    def __neg__(self): return self * -1.0
    def __sub__(self, other): return self + (-Tensor.wrap(other))
    def __rsub__(self, other): return Tensor.wrap(other) + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * Tensor.wrap(other) ** -1

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape)
        out._backward = _backward
        return out

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()

        def build(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


x = Tensor([[1.0, 2, 3], [4, 5, 6]])
b = Tensor([1.0, 1, 1])
L = (x * b).sum()
L.backward()
print(b.grad)        # [5. 7. 9.]
