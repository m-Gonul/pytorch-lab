import math
import random

TRUE = [i / 6 for i in range(1, 6)]


def walk_episode(rng):
    s, out = 3, []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        r = 1.0 if s2 == 6 else 0.0
        out.append((s, r, s2))
        if s2 in (0, 6):
            return out
        s = s2


def rms(V):
    return math.sqrt(sum((V[i] - TRUE[i - 1]) ** 2 for i in range(1, 6)) / 5)


def mc(episodes, alpha, V0=0.5):
    V = [0.0] + [V0] * 5 + [0.0]
    hist = []
    for ep in episodes:
        G = 0.0
        for s, r, _ in reversed(ep):
            G = r + G
            V[s] += alpha * (G - V[s])
        hist.append(rms(V))
    return V, hist


def td0(episodes, alpha, V0=0.5):
    V = [0.0] + [V0] * 5 + [0.0]
    hist = []
    for ep in episodes:
        for s, r, s2 in ep:
            V[s] += alpha * (r + V[s2] - V[s])
        hist.append(rms(V))
    return V, hist


rng = random.Random(0)
eps = [walk_episode(rng) for _ in range(100)]
print(rms(td0(eps, 0.1)[0]), rms(mc(eps, 0.02)[0]))
