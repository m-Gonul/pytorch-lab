import numpy as np


def _back(h_in, h_out, W, g):
    d = g * (1 - h_out ** 2)
    return h_in.T @ d, d @ W.T


def grads_full(x, Ws, gout):
    hs = [x]
    for W in Ws:
        hs.append(np.tanh(hs[-1] @ W))
    grads = [None] * len(Ws)
    g = gout
    for i in range(len(Ws) - 1, -1, -1):
        grads[i], g = _back(hs[i], hs[i + 1], Ws[i], g)
    return grads


def grads_ckpt(x, Ws, gout, k):
    n = len(Ws)
    starts = list(range(0, n, k))
    ends = {}
    h = x
    for i, W in enumerate(Ws):
        h = np.tanh(h @ W)
        if i == n - 1 or (i + 1) % k == 0:
            ends[i] = h
    peak = len(ends)
    grads = [None] * n
    g = gout
    for s in reversed(starts):
        e = min(s + k, n) - 1
        h_in = x if s == 0 else ends[s - 1]
        acts = [h_in]
        for i in range(s, e):
            acts.append(np.tanh(acts[-1] @ Ws[i]))
        acts.append(ends[e])
        peak = max(peak, len(ends) + (e - s))
        for i in range(e, s - 1, -1):
            grads[i], g = _back(acts[i - s], acts[i - s + 1], Ws[i], g)
        del ends[e]
    return grads, peak


rng = np.random.default_rng(0)
Ws = [rng.normal(size=(8, 8)) / 3 for _ in range(16)]
x, gout = rng.normal(size=(4, 8)), rng.normal(size=(4, 8))
g1 = grads_full(x, Ws, gout)
g2, peak = grads_ckpt(x, Ws, gout, 4)
print(max(np.abs(a - b).max() for a, b in zip(g1, g2)), peak)   # ~0, 7
