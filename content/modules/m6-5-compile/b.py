import math


class Graph:
    def __init__(self):
        self.nodes = []
        self.output = None

    def add(self, op, args):
        name = f"v{len(self.nodes)}"
        self.nodes.append((name, op, tuple(args)))
        return name


# ---- senin kodun ----

class Proxy:
    def __init__(self, graph, name):
        self.graph, self.name = graph, name

    def __add__(self, o):
        ...

    __radd__ = __add__

    def __sub__(self, o):
        ...

    def __mul__(self, o):
        ...

    __rmul__ = __mul__


def tanh(v):
    ...


def exp(v):
    ...


def trace(fn, n_inputs):
    ...


def run(graph, inputs):
    ...


def cse(graph):
    ...


def dce(graph):
    ...


def f(x, y):
    a = tanh(x * y)
    b = tanh(y * x)          # aynı ifade (değişmeli)
    unused = exp(x)          # kullanılmıyor
    return a + b * 2
g = trace(f, 2)
print(len(g.nodes), run(g, [0.3, 0.5]), f(0.3, 0.5))
g = dce(cse(g))
print(len(g.nodes), run(g, [0.3, 0.5]))
