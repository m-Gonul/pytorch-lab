import random
import numpy as np


def draw(rng):
    return min(rng.randint(1, 13), 10)


def _dealer(rng, up):
    cards = [up, draw(rng)]
    while True:
        s = sum(cards)
        t = s + 10 if 1 in cards and s + 10 <= 21 else s
        if t >= 17:
            return t
        cards.append(draw(rng))


def _episode(rng, policy):
    usable, total, up = rng.randrange(2), rng.randint(12, 21), rng.randint(1, 10)
    a = rng.randrange(2)
    traj = []
    while True:
        traj.append((usable, total, up, a))
        if a == 0:
            break
        total += draw(rng)
        if total > 21:
            if usable:
                total -= 10
                usable = 0
            else:
                return traj, -1
        a = policy[usable, total - 12, up - 1]
    d = _dealer(rng, up)
    return traj, (1 if d > 21 or total > d else (0 if total == d else -1))


def mc_es(n, seed=0):
    rng = random.Random(seed)
    Q = np.zeros((2, 10, 10, 2))
    C = np.zeros((2, 10, 10, 2))
    policy = np.ones((2, 10, 10), dtype=int)
    policy[:, 8:, :] = 0
    for _ in range(n):
        traj, r = _episode(rng, policy)
        for u, t, d, a in traj:
            i = (u, t - 12, d - 1)
            C[i + (a,)] += 1
            Q[i + (a,)] += (r - Q[i + (a,)]) / C[i + (a,)]
            policy[i] = int(Q[i + (1,)] > Q[i + (0,)])
    return Q, policy


Q, pol = mc_es(20000)
print(pol[0, ::-1])
