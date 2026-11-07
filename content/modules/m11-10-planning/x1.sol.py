import math
import random

H, W = 6, 9
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
START, GOAL = (5, 3), (0, 8)
WALL_A = {(3, c) for c in range(1, 9)}
WALL_B = {(3, c) for c in range(1, 8)}


def _step(s, a, walls):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    if not (0 <= r < H and 0 <= c < W) or (r, c) in walls:
        r, c = s
    return (r, c), (1.0 if (r, c) == GOAL else 0.0), (r, c) == GOAL


def run(plus, n_steps=6000, change=3000, kappa=1e-3, seed=0, n_plan=50):
    rng = random.Random(seed)
    Q, model, last, keys, seen = {}, {}, {}, [], set()

    def q(s):
        return Q.setdefault(s, [0.0] * 4)

    def put(k, v):
        if k not in model:
            keys.append(k)
        model[k] = v

    s, cum, out = START, 0.0, []
    for t in range(n_steps):
        walls = WALL_A if t < change else WALL_B
        qs = q(s)
        if rng.random() < 0.1:
            a = rng.randrange(4)
        else:
            m = max(qs)
            a = rng.choice([i for i in range(4) if qs[i] == m])
        s2, r, done = _step(s, a, walls)
        qs[a] += r + (0.0 if done else 0.95 * max(q(s2))) - qs[a]
        if plus and s not in seen:
            for b in range(4):
                if b != a and (s, b) not in model:
                    put((s, b), (0.0, s, False))
        seen.add(s)
        put((s, a), (r, s2, done))
        last[(s, a)] = t
        for _ in range(n_plan):
            ps, pa = keys[rng.randrange(len(keys))]
            pr, ps2, pd = model[(ps, pa)]
            if plus:
                pr += kappa * math.sqrt(t - last.get((ps, pa), 0))
            qq = q(ps)
            qq[pa] += pr + (0.0 if pd else 0.95 * max(q(ps2))) - qq[pa]
        cum += r
        out.append(cum)
        s = START if done else s2
    return out


print(run(True, 2000)[-1])
