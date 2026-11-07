MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def grid_step(s, a):
    r, c = divmod(s, 4)
    dr, dc = MOVES[a]
    rr, cc = min(max(r + dr, 0), 3), min(max(c + dc, 0), 3)
    return rr * 4 + cc, -1.0


def _terminal(s):
    return s in (0, 15)


def evaluate(pi, gamma=1.0, theta=1e-10):
    V = [0.0] * 16
    sweeps = 0
    while True:
        delta = 0.0
        sweeps += 1
        for s in range(16):
            if _terminal(s):
                continue
            v = 0.0
            for a in range(4):
                s2, r = grid_step(s, a)
                v += pi[s][a] * (r + gamma * V[s2])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            return V, sweeps


def improve(V, gamma=1.0):
    pi = []
    for s in range(16):
        q = []
        for a in range(4):
            s2, r = grid_step(s, a)
            q.append(r + gamma * V[s2])
        best = q.index(max(q))
        pi.append([1.0 if a == best else 0.0 for a in range(4)])
    return pi


def policy_iteration(gamma=1.0):
    pi = [[0.25] * 4 for _ in range(16)]
    n = 0
    while True:
        V, _ = evaluate(pi, gamma)
        new = improve(V, gamma)
        n += 1
        if new == pi:
            return V, pi, n
        pi = new


V, sweeps = evaluate([[0.25] * 4 for _ in range(16)])
print([round(v) for v in V])
