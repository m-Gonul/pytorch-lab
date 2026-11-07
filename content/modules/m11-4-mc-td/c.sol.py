import numpy as np


def _flat(episodes):
    s, r, s2, G = [], [], [], []
    for ep in episodes:
        st = [x for x, _ in ep]; rw = [y for _, y in ep]
        s += st; r += rw; s2 += st[1:] + [-1]
        G += list(np.cumsum(rw[::-1])[::-1])
    return np.array(s), np.array(r, float), np.array(s2), np.array(G, float)


def batch_mc(episodes, n_states):
    s, _, _, G = _flat(episodes)
    cnt = np.bincount(s, minlength=n_states)
    return np.bincount(s, weights=G, minlength=n_states) / np.maximum(cnt, 1)


def batch_td(episodes, n_states, alpha=0.01, tol=1e-10):
    s, r, s2, _ = _flat(episodes)
    V = np.zeros(n_states)
    while True:
        nxt = np.where(s2 >= 0, V[np.maximum(s2, 0)], 0.0)
        d = np.zeros(n_states)
        np.add.at(d, s, r + nxt - V[s])
        V = V + alpha * d
        if np.abs(alpha * d).max() < tol:
            return V


def certainty_equivalence(episodes, n_states):
    s, r, s2, _ = _flat(episodes)
    cnt = np.bincount(s, minlength=n_states).astype(float)
    P = np.zeros((n_states, n_states))
    ok = s2 >= 0
    np.add.at(P, (s[ok], s2[ok]), 1)
    P /= np.maximum(cnt, 1)[:, None]
    rbar = np.bincount(s, weights=r, minlength=n_states) / np.maximum(cnt, 1)
    return np.linalg.solve(np.eye(n_states) - P, rbar)


AB = [[(0, 0.0), (1, 0.0)]] + [[(1, 1.0)]] * 6 + [[(1, 0.0)]]
print(batch_mc(AB, 2), batch_td(AB, 2))
