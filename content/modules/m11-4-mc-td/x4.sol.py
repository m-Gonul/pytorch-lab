import numpy as np


def chain_step(s, a, N):
    """Zincir: a = 0 sola, 1 sağa. Solda (s = 0) sola gitmek 0.05 ödül, sağ uçta (s = N−1) sağa gitmek 1 ödül."""
    if a == 0:
        return max(s - 1, 0), (0.05 if s == 0 else 0.0)
    return min(s + 1, N - 1), (1.0 if s == N - 1 else 0.0)


def select(q, method, t, rng, eps=0.1, tau=0.1):
    if method == "greedy":
        best = np.flatnonzero(q == q.max())
        return int(rng.choice(best))
    if method in ("eps", "eps-decay"):
        e = eps if method == "eps" else max(0.05, 1.0 - t / 200)
        if rng.random() < e:
            return int(rng.integers(len(q)))
        return select(q, "greedy", t, rng)
    if method == "boltzmann":
        z = (q - q.max()) / tau
        p = np.exp(z) / np.exp(z).sum()
        return int(rng.choice(len(q), p=p))
    raise ValueError(method)


def q_learning(N, episodes, method, q0, seed, alpha=0.2, gamma=0.95):
    rng = np.random.default_rng(seed)
    Q = np.full((N, 2), float(q0))
    returns, first = [], None
    for ep in range(episodes):
        s, G = 1, 0.0
        for _ in range(N + 6):
            a = select(Q[s], method, ep, rng)
            s2, r = chain_step(s, a, N)
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
            G += r
            if r == 1.0 and first is None:
                first = ep
            s = s2
        returns.append(G)
    return np.array(returns), first
