import random

H, W = 6, 9
WALLS = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}
START, GOAL = (2, 0), (0, 8)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def maze_step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    if not (0 <= r < H and 0 <= c < W) or (r, c) in WALLS:
        r, c = s
    return (r, c), (1.0 if (r, c) == GOAL else 0.0), (r, c) == GOAL


def dyna_q(n_eps, n_plan, seed=0, alpha=0.1, gamma=0.95, eps=0.1):
    rng = random.Random(seed)
    Q, model, keys, steps = {}, {}, [], []

    def q(s):
        return Q.setdefault(s, [0.0] * 4)

    for _ in range(n_eps):
        s, k = START, 0
        while True:
            qs = q(s)
            if rng.random() < eps:
                a = rng.randrange(4)
            else:
                m = max(qs)
                a = rng.choice([i for i in range(4) if qs[i] == m])
            s2, r, done = maze_step(s, a)
            k += 1
            qs[a] += alpha * (r + (0.0 if done else gamma * max(q(s2))) - qs[a])
            if (s, a) not in model:
                keys.append((s, a))
            model[(s, a)] = (r, s2, done)
            for _ in range(n_plan):
                ps, pa = keys[rng.randrange(len(keys))]
                pr, ps2, pd = model[(ps, pa)]
                qq = q(ps)
                qq[pa] += alpha * (pr + (0.0 if pd else gamma * max(q(ps2))) - qq[pa])
            if done:
                break
            s = s2
        steps.append(k)
    return Q, steps


print(dyna_q(20, 5)[1])
