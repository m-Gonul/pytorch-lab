def forward(graph, inputs):
    ...


def toposort(graph, out):
    ...


def backward(graph, vals, out, hooks=None):
    ...


g = {"a": ("leaf", []), "b": ("leaf", []), "c": ("mul", ["a", "b"]), "d": ("add", ["c", "a"]), "L": ("square", ["d"])}
vals = forward(g, {"a": 2.0, "b": 3.0})
print(backward(g, vals, "L"))                                # {'a': 64, 'b': 32}
print(backward(g, vals, "L", hooks={"c": lambda x: 2 * x}))  # {'a': 112, 'b': 64}
