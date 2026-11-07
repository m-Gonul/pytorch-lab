import numpy as np


def evaluate(graph, inputs):
    ...


def fuse(graph, outputs):
    ...


def bytes_moved(graph, groups, nbytes, outputs):
    ...


G = [("x", "input", ()), ("a", "input", ()), ("b", "input", ()),
     ("t1", "mul", ("x", "a")), ("t2", "add", ("t1", "b")), ("y", "tanh", ("t2",))]
N = 10 ** 6
nb = {"x": 4 * N, "a": 0, "b": 0, "t1": 4 * N, "t2": 4 * N, "y": 4 * N}
groups = fuse(G, ["y"])
print(groups)                                                   # [['t1', 't2', 'y']]
print(bytes_moved(G, [["t1"], ["t2"], ["y"]], nb, ["y"]), bytes_moved(G, groups, nb, ["y"]))   # 24e6, 8e6
