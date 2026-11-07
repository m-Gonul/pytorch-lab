def forward(graph, inputs):
    vals = {}
    for n in toposort_all(graph):
        op, ins = graph[n]
        if op == "leaf":
            vals[n] = inputs[n]
        elif op == "add":
            vals[n] = vals[ins[0]] + vals[ins[1]]
        elif op == "mul":
            vals[n] = vals[ins[0]] * vals[ins[1]]
        elif op == "square":
            vals[n] = vals[ins[0]] ** 2
    return vals


def toposort_all(graph):
    order, seen = [], set()
    for n in graph:
        _dfs(graph, n, seen, order)
    return order


def _dfs(graph, n, seen, order):
    if n in seen:
        return
    seen.add(n)
    for i in graph[n][1]:
        _dfs(graph, i, seen, order)
    order.append(n)


def toposort(graph, out):
    order = []
    _dfs(graph, out, set(), order)
    return order


def backward(graph, vals, out, hooks=None):
    hooks = hooks or {}
    g = {out: 1.0}
    leaves = {}
    for n in reversed(toposort(graph, out)):
        if n not in g:
            continue
        gn = g[n]
        if n in hooks:
            gn = hooks[n](gn)
        op, ins = graph[n]
        if op == "leaf":
            leaves[n] = gn
            continue
        if op == "add":
            parts = [gn, gn]
        elif op == "mul":
            parts = [gn * vals[ins[1]], gn * vals[ins[0]]]
        else:
            parts = [gn * 2 * vals[ins[0]]]
        for i, p in zip(ins, parts):
            g[i] = g.get(i, 0.0) + p
    return leaves


g = {"a": ("leaf", []), "b": ("leaf", []), "c": ("mul", ["a", "b"]), "d": ("add", ["c", "a"]), "L": ("square", ["d"])}
vals = forward(g, {"a": 2.0, "b": 3.0})
print(backward(g, vals, "L"))                                # {'a': 64, 'b': 32}
print(backward(g, vals, "L", hooks={"c": lambda x: 2 * x}))  # {'a': 112, 'b': 64}
