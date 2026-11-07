import numpy as np


def grad_norms(X, Ws, residual):
    """→ [‖g_0‖, …, ‖g_L‖]. Yalnız katman döngüleri."""
    h, ts = X, []
    for W in Ws:
        t = np.tanh(h @ W)
        ts.append(t)
        h = h + t if residual else t
    g = np.ones_like(h)
    norms = [np.linalg.norm(g)]
    for W, t in zip(reversed(Ws), reversed(ts)):
        local = (g * (1 - t ** 2)) @ W.T
        g = g + local if residual else local
        norms.append(np.linalg.norm(g))
    return norms[::-1]


rng = np.random.default_rng(0)
D, L = 64, 20
X = rng.standard_normal((32, D))
Ws = [rng.standard_normal((D, D)) * 0.5 / np.sqrt(D) for _ in range(L)]
print(grad_norms(X, Ws, False)[0], grad_norms(X, Ws, True)[0])
