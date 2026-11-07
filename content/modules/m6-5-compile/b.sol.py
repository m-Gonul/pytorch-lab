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

def _arg(o):
    return o.name if isinstance(o, Proxy) else o


class Proxy:
    def __init__(self, graph, name):
        self.graph, self.name = graph, name

    def _op(self, op, *args):
        return Proxy(self.graph, self.graph.add(op, [_arg(a) for a in args]))

    def __add__(self, o):
        return self._op("add", self, o)

    __radd__ = __add__

    def __sub__(self, o):
        return self._op("sub", self, o)

    def __mul__(self, o):
        return self._op("mul", self, o)

    __rmul__ = __mul__


def tanh(v):
    return v._op("tanh", v) if isinstance(v, Proxy) else math.tanh(v)


def exp(v):
    return v._op("exp", v) if isinstance(v, Proxy) else math.exp(v)


def trace(fn, n_inputs):
    g = Graph()
    ins = [Proxy(g, g.add("input", [i])) for i in range(n_inputs)]
    out = fn(*ins)
    g.output = out.name
    return g


def run(graph, inputs):
    val = {}
    ops = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
           "tanh": math.tanh, "exp": math.exp}
    for name, op, args in graph.nodes:
        if op == "input":
            val[name] = inputs[args[0]]
        else:
            val[name] = ops[op](*[val[a] if isinstance(a, str) else a for a in args])
    return val[graph.output]


def cse(graph):
    g = Graph()
    seen, rename = {}, {}
    for name, op, args in graph.nodes:
        args = tuple(rename.get(a, a) if isinstance(a, str) else a for a in args)
        key = (op, tuple(sorted(args, key=repr)) if op in ("add", "mul") else args)
        if op != "input" and key in seen:
            rename[name] = seen[key]
            continue
        seen[key] = name
        rename[name] = name
        g.nodes.append((name, op, args))
    g.output = rename[graph.output]
    return g


def dce(graph):
    live = {graph.output}
    for name, op, args in reversed(graph.nodes):
        if name in live:
            live.update(a for a in args if isinstance(a, str))
    g = Graph()
    g.nodes = [n for n in graph.nodes if n[0] in live or n[1] == "input"]
    g.output = graph.output
    return g


def f(x, y):
    a = tanh(x * y)
    b = tanh(y * x)          # aynı ifade (değişmeli)
    unused = exp(x)          # kullanılmıyor
    return a + b * 2
g = trace(f, 2)
print(len(g.nodes), run(g, [0.3, 0.5]), f(0.3, 0.5))
g = dce(cse(g))
print(len(g.nodes), run(g, [0.3, 0.5]))
