import math
import random

P = 0.3


def _J(p):
    return (2 * p - 4) / (p * (1 - p))


def _V(p):
    v1 = (p - 3) / (p * (1 - p))
    return [_J(p), v1, -1 + (1 - p) * v1]


def true_grad():
    h = 1e-6
    return (_J(P + h) - _J(P - h)) / (2 * h) * P * (1 - P)


def grad_samples(n, kind, seed=0):
    rng = random.Random(seed)
    V = _V(P)
    out = []
    for _ in range(n):
        s, traj = 0, []
        while s != 3:
            right = rng.random() < P
            traj.append((s, right))
            s2 = (s - 1 if right else s + 1) if s == 1 else (s + 1 if right else max(s - 1, 0))
            s = s2
        T = len(traj)
        g = 0.0
        for t, (s, right) in enumerate(traj):
            G = -(T - t)
            b = 0.0 if kind == "G" else (V[0] if kind == "b" else V[s])
            g += (G - b) * ((1 - P) if right else -P)
        out.append(g)
    return out


print(true_grad(), sum(grad_samples(2000, "V")) / 2000)
