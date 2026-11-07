import heapq
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


def greedy_len(Q):
    s = START
    for k in range(1, 60):
        qs = Q.get(s, [0.0] * 4)
        s, _, done = maze_step(s, qs.index(max(qs)))
        if done:
            return k
    return 999


def _act(Q, s, rng):
    qs = Q.setdefault(s, [0.0] * 4)
    if rng.random() < 0.1:
        return rng.randrange(4)
    m = max(qs)
    return rng.choice([i for i in range(4) if qs[i] == m])


def _target(Q, r, s2, done):
    return r + (0.0 if done else 0.95 * max(Q.setdefault(s2, [0.0] * 4)))


def prioritized(n_plan, theta=1e-4, seed=0, max_steps=20000, alpha=0.5):
    rng = random.Random(seed)
    Q, model, pred, pq = {}, {}, {}, []
    s, updates = START, 0
    for _ in range(max_steps):
        a = _act(Q, s, rng)
        s2, r, done = maze_step(s, a)
        model[(s, a)] = (r, s2, done)
        pred.setdefault(s2, set()).add((s, a))
        p = abs(_target(Q, r, s2, done) - Q[s][a])
        if p > theta:
            heapq.heappush(pq, (-p, (s, a)))
        for _ in range(n_plan):
            if not pq:
                break
            _, (ps, pa) = heapq.heappop(pq)
            pr, ps2, pd = model[(ps, pa)]
            Q[ps][pa] += alpha * (_target(Q, pr, ps2, pd) - Q[ps][pa])
            updates += 1
            for (bs, ba) in pred.get(ps, ()):
                br, _, bd = model[(bs, ba)]
                bp = abs(_target(Q, br, ps, bd) - Q.setdefault(bs, [0.0] * 4)[ba])
                if bp > theta:
                    heapq.heappush(pq, (-bp, (bs, ba)))
        if greedy_len(Q) == 14:
            return updates
        s = START if done else s2
    return updates


def dyna_updates(n_plan, seed=0, max_steps=20000, alpha=0.5):
    rng = random.Random(seed)
    Q, model, keys = {}, {}, []
    s, updates = START, 0
    for _ in range(max_steps):
        a = _act(Q, s, rng)
        s2, r, done = maze_step(s, a)
        Q[s][a] += alpha * (_target(Q, r, s2, done) - Q[s][a])
        updates += 1
        if (s, a) not in model:
            keys.append((s, a))
        model[(s, a)] = (r, s2, done)
        for _ in range(n_plan):
            ps, pa = keys[rng.randrange(len(keys))]
            pr, ps2, pd = model[(ps, pa)]
            Q[ps][pa] += alpha * (_target(Q, pr, ps2, pd) - Q[ps][pa])
            updates += 1
        if greedy_len(Q) == 14:
            return updates
        s = START if done else s2
    return updates


print(prioritized(5), dyna_updates(5))
