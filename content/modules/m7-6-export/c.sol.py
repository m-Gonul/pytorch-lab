import numpy as np

OPS = {
    "Gemm": lambda x, W, b: x @ W.T + b,
    "Relu": lambda x: np.maximum(x, 0),
    "Add": np.add,
    "Mul": np.multiply,
    "Transpose": lambda x: x.T,
    "Softmax": lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True),
}


def run_graph(nodes, inits, feeds):
    vals = dict(inits)
    vals.update(feeds)
    for n in nodes:
        vals[n["output"]] = OPS[n["op"]](*[vals[i] for i in n["inputs"]])
    return vals


def fold_constants(nodes, inits):
    inits = dict(inits)
    keep = []
    for n in nodes:
        if all(i in inits for i in n["inputs"]):
            inits[n["output"]] = OPS[n["op"]](*[inits[i] for i in n["inputs"]])
        else:
            keep.append(n)
    return keep, inits


def _bcast(a, b):
    a, b = tuple(a), tuple(b)
    n = max(len(a), len(b))
    a, b = (1,) * (n - len(a)) + a, (1,) * (n - len(b)) + b
    return tuple(y if x == 1 else x for x, y in zip(a, b))


def infer_shapes(nodes, inits, input_shapes):
    sh = {k: tuple(np.shape(v)) for k, v in inits.items()}
    sh.update({k: tuple(v) for k, v in input_shapes.items()})
    for n in nodes:
        ins = [sh[i] for i in n["inputs"]]
        op = n["op"]
        if op == "Gemm":
            out = (ins[0][0], ins[1][0])
        elif op == "Transpose":
            out = tuple(reversed(ins[0]))
        elif op in ("Add", "Mul"):
            out = _bcast(ins[0], ins[1])
        else:
            out = ins[0]
        sh[n["output"]] = out
    return sh


rng = np.random.default_rng(0)
inits = {"W1": rng.normal(size=(8, 4)), "b1": np.zeros(8), "two": np.array(2.0), "W2": rng.normal(size=(8, 3)), "b2": np.zeros(3)}
nodes = [
    {"op": "Gemm", "inputs": ["x", "W1", "b1"], "output": "h"},
    {"op": "Relu", "inputs": ["h"], "output": "r"},
    {"op": "Mul", "inputs": ["W2", "two"], "output": "W2s"},
    {"op": "Transpose", "inputs": ["W2s"], "output": "W2t"},
    {"op": "Gemm", "inputs": ["r", "W2t", "b2"], "output": "y"},
    {"op": "Softmax", "inputs": ["y"], "output": "p"},
]
print(run_graph(nodes, inits, {"x": rng.normal(size=(5, 4))})["p"].sum(1))
n2, i2 = fold_constants(nodes, inits)
print([n["op"] for n in n2])
print(infer_shapes(nodes, inits, {"x": ("N", 4)}))
