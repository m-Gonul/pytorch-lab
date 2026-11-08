import numpy as np


def init_std(kind, fan_in, fan_out):
    v = {"glorot": 2 / (fan_in + fan_out), "he": 2 / fan_in, "lecun": 1 / fan_in}[kind]
    return float(np.sqrt(v))


def uniform_limit(kind, fan_in, fan_out):
    return float(np.sqrt(3) * init_std(kind, fan_in, fan_out))


def variance_flow(sizes, kind, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0, init_std(kind, a, b), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    H = rng.normal(size=(n, sizes[0]))
    fwd = []
    for W in Ws:
        H2 = H @ W
        fwd.append(float(H2.var() / H.var()))
        H = H2
    G = rng.normal(size=(n, sizes[-1]))
    bwd = []
    for W in Ws[::-1]:
        G2 = G @ W.T
        bwd.append(float(G2.var() / G.var()))
        G = G2
    return fwd, bwd[::-1]
