import random

WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def windy_step(r, c, a):
    dr, dc = MOVES[a]
    r2 = min(max(r + dr - WIND[c], 0), 6)
    c2 = min(max(c + dc, 0), 9)
    return r2, c2


def _eg(Q, s, rng, eps=0.1):
    if rng.random() < eps:
        return rng.randrange(4)
    m = max(Q[s])
    return rng.choice([a for a in range(4) if Q[s][a] == m])


def sarsa_windy(n_steps, seed=0):
    rng = random.Random(seed)
    Q = {(r, c): [0.0] * 4 for r in range(7) for c in range(10)}
    s = (3, 0)
    a = _eg(Q, s, rng)
    done_eps, hist = 0, []
    for _ in range(n_steps):
        s2 = windy_step(*s, a)
        a2 = _eg(Q, s2, rng)
        goal = s2 == (3, 7)
        Q[s][a] += 0.5 * (-1 + (0.0 if goal else Q[s2][a2]) - Q[s][a])
        if goal:
            done_eps += 1
            s = (3, 0)
            a = _eg(Q, s, rng)
        else:
            s, a = s2, a2
        hist.append(done_eps)
    return Q, hist


def greedy_len(Q):
    s = (3, 0)
    for k in range(1, 101):
        s = windy_step(*s, Q[s].index(max(Q[s])))
        if s == (3, 7):
            return k
    return 100


Q, ep = sarsa_windy(8000)
print(ep[-1], greedy_len(Q))
