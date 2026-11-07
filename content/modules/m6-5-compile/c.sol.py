import numpy as np

EW = {"add", "mul", "tanh", "exp"}


def evaluate(graph, inputs):
    ops = {"add": np.add, "mul": np.multiply, "tanh": np.tanh, "exp": np.exp, "sum": np.sum}
    val = {}
    for name, op, args in graph:
        val[name] = inputs[name] if op == "input" else ops[op](*[val[a] for a in args])
    return val


def _users(graph):
    users = {}
    for name, _, args in graph:
        for a in args:
            users.setdefault(a, []).append(name)
    return users


def fuse(graph, outputs):
    ops = {n: op for n, op, _ in graph}
    users = _users(graph)
    group_of, groups = {}, []
    for name, op, args in graph:
        if op == "input":
            continue
        target = None
        if op in EW:
            for a in args:
                if ops[a] in EW and len(users.get(a, [])) == 1 and a not in outputs:
                    target = group_of[a]
                    break
        if target is None:
            groups.append([name])
            group_of[name] = len(groups) - 1
        else:
            groups[target].append(name)
            group_of[name] = target
    return groups


def bytes_moved(graph, groups, nbytes, outputs):
    args = {n: a for n, _, a in graph}
    users = _users(graph)
    tot = 0
    for grp in groups:
        inside = set(grp)
        reads = {a for n in grp for a in args[n] if a not in inside}
        writes = {n for n in grp if n in outputs or any(u not in inside for u in users.get(n, []))}
        tot += sum(nbytes[a] for a in reads) + sum(nbytes[n] for n in writes)
    return tot


G = [("x", "input", ()), ("a", "input", ()), ("b", "input", ()),
     ("t1", "mul", ("x", "a")), ("t2", "add", ("t1", "b")), ("y", "tanh", ("t2",))]
N = 10 ** 6
nb = {"x": 4 * N, "a": 0, "b": 0, "t1": 4 * N, "t2": 4 * N, "y": 4 * N}
groups = fuse(G, ["y"])
print(groups)                                                   # [['t1', 't2', 'y']]
print(bytes_moved(G, [["t1"], ["t2"], ["y"]], nb, ["y"]), bytes_moved(G, groups, nb, ["y"]))   # 24e6, 8e6
