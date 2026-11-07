import numpy as np


def run_graph(nodes, inits, feeds):
    ...


def fold_constants(nodes, inits):
    ...


def infer_shapes(nodes, inits, input_shapes):
    ...


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
