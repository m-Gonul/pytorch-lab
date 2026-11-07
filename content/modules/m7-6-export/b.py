import json


def to_json(layers):
    ...


def from_json(s):
    ...


def run(layers, x):
    ...


class Rec:
    def __init__(self, values, graph):
        self.values, self.graph = list(values), graph

    def __mul__(self, c):
        ...

    __rmul__ = __mul__

    def __neg__(self):
        ...

    def sum(self):
        ...


def trace(fn, x):
    ...


def replay(graph, x):
    ...


f = lambda x: x * 2 if x.sum() > 0 else -x
g = trace(f, [1.0, 2.0])
print(g, replay(g, [-1.0, -1.0]))   # [('mul', 2)] [-2, -2]  (yanlış dal donmuş)
