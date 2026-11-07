import random


def walk_episode(rng, n=1000, jump=100):
    s, out = n // 2, []
    while True:
        k = rng.randint(1, jump)
        s2 = s + k if rng.random() < 0.5 else s - k
        if s2 < 1:
            out.append((s, -1.0, None))
            return out
        if s2 > n:
            out.append((s, 1.0, None))
            return out
        out.append((s, 0.0, s2))
        s = s2


def group(s, k=10):
    return (s - 1) * k // 1000


def gradient_mc(episodes, alpha, k=10):
    w = [0.0] * k
    for ep in episodes:
        G = sum(r for _, r, _ in ep)
        for s, _, _ in ep:
            g = group(s, k)
            w[g] += alpha * (G - w[g])
    return w


def semi_gradient_td(episodes, alpha, k=10):
    w = [0.0] * k
    for ep in episodes:
        for s, r, s2 in ep:
            g = group(s, k)
            v2 = 0.0 if s2 is None else w[group(s2, k)]
            w[g] += alpha * (r + v2 - w[g])
    return w


rng = random.Random(0)
eps = [walk_episode(rng) for _ in range(200)]
print(semi_gradient_td(eps, 2e-3))
