import numpy as np


def unbroadcast(grad, shape):
    ...


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

    # __mul__, __matmul__, __neg__, __sub__, __pow__, __truediv__, __radd__, __rmul__, __rsub__
    # sum, mean, relu, exp, log, backward
    ...


x = Tensor([[1.0, 2, 3], [4, 5, 6]])
b = Tensor([1.0, 1, 1])
L = (x * b).sum()
L.backward()
print(b.grad)        # [5. 7. 9.]
