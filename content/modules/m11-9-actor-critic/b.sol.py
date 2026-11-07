import math
import random


def corridor_step(s, right):
    if s == 1:
        right = not right
    s2 = s + 1 if right else max(s - 1, 0)
    return s2, -1.0, s2 == 3


def corridor_value(p):
    return (2 * p - 4) / (p * (1 - p))


def _p(theta):
    return 1 / (1 + math.exp(-(theta[0] - theta[1])))


def reinforce(n_eps, alpha, seed=0, baseline=False, alpha_w=0.0, max_steps=1000):
    rng = random.Random(seed)
    theta = [math.log(0.05 / 0.95), 0.0]
    w = 0.0
    hist = []
    for _ in range(n_eps):
        s, traj = 0, []
        for _ in range(max_steps):
            right = rng.random() < _p(theta)
            s, r, done = corridor_step(s, right)
            traj.append((right, r))
            if done:
                break
        G, Gs = 0.0, []
        for _, r in reversed(traj):
            G += r
            Gs.append(G)
        Gs.reverse()
        for (right, _), G in zip(traj, Gs):
            delta = G - w if baseline else G
            if baseline:
                w += alpha_w * delta
            p = _p(theta)
            g0 = (1 - p) if right else -p
            theta[0] += alpha * delta * g0
            theta[1] -= alpha * delta * g0
        hist.append(_p(theta))
    return theta, hist


theta, ps = reinforce(200, 2 ** -10, baseline=True, alpha_w=2 ** -6)
print(ps[-1], corridor_value(ps[-1]))
