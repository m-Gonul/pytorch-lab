import math
import random


def mc_step(p, v, a):
    v = min(max(v + 0.001 * (a - 1) - 0.0025 * math.cos(3 * p), -0.07), 0.07)
    p = p + v
    if p < -1.2:
        p, v = -1.2, 0.0
    return p, v, p >= 0.5


def tiles(p, v, a, n_tilings=8, n=8):
    ps = (p + 1.2) / 1.7 * n
    vs = (v + 0.07) / 0.14 * n
    out = []
    for t in range(n_tilings):
        i = int(ps + t / n_tilings)
        j = int(vs + (3 * t % n_tilings) / n_tilings)
        out.append(((t * (n + 1) + i) * (n + 1) + j) * 3 + a)
    return out


def sarsa_mc(n_eps, alpha=0.5, seed=0, n_tilings=8, max_steps=5000):
    rng = random.Random(seed)
    w = [0.0] * (n_tilings * 9 * 9 * 3)
    step = alpha / n_tilings

    def q(p, v, a):
        return sum(w[i] for i in tiles(p, v, a, n_tilings))

    def act(p, v):
        qs = [q(p, v, a) for a in range(3)]
        m = max(qs)
        return rng.choice([a for a in range(3) if qs[a] == m])

    lens = []
    for _ in range(n_eps):
        p, v = rng.uniform(-0.6, -0.4), 0.0
        a = act(p, v)
        steps = 0
        while True:
            p2, v2, done = mc_step(p, v, a)
            steps += 1
            F = tiles(p, v, a, n_tilings)
            qsa = sum(w[i] for i in F)
            if done or steps >= max_steps:
                target = -1.0
            else:
                a2 = act(p2, v2)
                target = -1.0 + q(p2, v2, a2)
            for i in F:
                w[i] += step * (target - qsa)
            if done or steps >= max_steps:
                break
            p, v, a = p2, v2, a2
        lens.append(steps)
    return w, lens


w, lens = sarsa_mc(20)
print(lens)
