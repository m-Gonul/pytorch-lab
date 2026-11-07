import numpy as np


def h_star(P, R, gamma, V):
    return (R + gamma * np.einsum("ast,t->sa", P, V)).max(1)


def find_l2_counterexample(trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    best = (0.0, None)
    for _ in range(trials):
        S, A = 3, 2
        P = rng.random((A, S, S)) ** 4; P /= P.sum(-1, keepdims=True)
        R = rng.normal(size=(S, A)); g = rng.uniform(0.5, 0.99)
        V, U = rng.normal(size=S) * 5, rng.normal(size=S) * 5
        r = np.linalg.norm(h_star(P, R, g, V) - h_star(P, R, g, U)) / np.linalg.norm(V - U)
        if r / g > best[0]:
            best = (r / g, (r, g))
    return best[1]


print(find_l2_counterexample(200))
