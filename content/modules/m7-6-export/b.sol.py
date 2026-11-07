import json


def to_json(layers):
    return json.dumps(layers)


def from_json(s):
    return json.loads(s)


def run(layers, x):
    for L in layers:
        if L["op"] == "linear":
            x = [sum(w * v for w, v in zip(row, x)) + b for row, b in zip(L["W"], L["b"])]
        elif L["op"] == "relu":
            x = [max(0.0, v) for v in x]
    return x


class Rec:
    def __init__(self, values, graph):
        self.values, self.graph = list(values), graph

    def __mul__(self, c):
        self.graph.append(("mul", c))
        return Rec([v * c for v in self.values], self.graph)

    __rmul__ = __mul__

    def __neg__(self):
        self.graph.append(("neg",))
        return Rec([-v for v in self.values], self.graph)

    def sum(self):
        return sum(self.values)


def trace(fn, x):
    graph = []
    fn(Rec(x, graph))
    return graph


def replay(graph, x):
    x = list(x)
    for op in graph:
        if op[0] == "mul":
            x = [v * op[1] for v in x]
        elif op[0] == "neg":
            x = [-v for v in x]
    return x


f = lambda x: x * 2 if x.sum() > 0 else -x
g = trace(f, [1.0, 2.0])
print(g, replay(g, [-1.0, -1.0]))   # [('mul', 2)] [-2, -2]  (yanlış dal donmuş)
