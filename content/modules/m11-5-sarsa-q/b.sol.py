import random
H, W = 4, 12
START, GOAL = 36, 47
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def cliff_step(s, a):
    r, c = divmod(s, W)
    dr, dc = MOVES[a]
    r = min(max(r + dr, 0), H - 1)
    c = min(max(c + dc, 0), W - 1)
    if r == 3 and 1 <= c <= 10:
        return START, -100.0, False
    s2 = r * W + c
    return s2, -1.0, s2 == GOAL


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return rng.randrange(4)
    q = Q[s]
    m = max(q)
    return rng.choice([a for a in range(4) if q[a] == m])


def _run(n_eps, alpha, eps, seed, on_policy):
    rng = random.Random(seed)
    Q = [[0.0] * 4 for _ in range(H * W)]
    rets = []
    for _ in range(n_eps):
        s = START
        a = eps_greedy(Q, s, eps, rng)
        G = 0.0
        while True:
            s2, r, done = cliff_step(s, a)
            G += r
            a2 = eps_greedy(Q, s2, eps, rng)
            nxt = 0.0 if done else (Q[s2][a2] if on_policy else max(Q[s2]))
            Q[s][a] += alpha * (r + nxt - Q[s][a])
            if done:
                break
            s, a = s2, a2
        rets.append(G)
    return Q, rets


def sarsa(n_eps, alpha=0.5, eps=0.1, seed=0):
    return _run(n_eps, alpha, eps, seed, True)


def qlearning(n_eps, alpha=0.5, eps=0.1, seed=0):
    return _run(n_eps, alpha, eps, seed, False)


def greedy_return(Q, max_steps=100):
    s, G = START, 0.0
    for _ in range(max_steps):
        q = Q[s]
        s, r, done = cliff_step(s, q.index(max(q)))
        G += r
        if done:
            break
    return G


print(greedy_return(qlearning(300)[0]))
