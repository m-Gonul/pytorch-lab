import math
import numpy as np

LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
BOARD = [1, 0, 0, -1, 1, 0, 0, -1, 0]
WIN = 8


def _term(b):
    for l in LINES:
        s = b[l[0]] + b[l[1]] + b[l[2]]
        if s == 3 or s == -3:
            return s // 3
    return 0 if all(b) else None


def _prior(b):
    legal = [i for i in range(9) if b[i] == 0]
    p = np.zeros(9)
    for i in legal:
        p[i] = 0.001 if (b is BOARD and i == WIN) else 1.0
    return p / p.sum()


def root_visits(n_sims, noise, rng, c=1.5):
    P = _prior(BOARD)
    if noise:
        legal = np.array([v == 0 for v in BOARD])
        eta = np.zeros(9)
        eta[legal] = rng.dirichlet([0.3] * int(legal.sum()))
        P = 0.75 * P + 0.25 * eta
    N, W = np.zeros(9), np.zeros(9)
    legal = np.array([v == 0 for v in BOARD])
    for _ in range(n_sims):
        Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
        U = c * P * math.sqrt(N.sum() + 1) / (1 + N)
        a = int(np.argmax(np.where(legal, Q + U, -1e9)))
        b2 = BOARD[:]
        b2[a] = 1
        t = _term(b2)
        v = 1.0 if t == 1 else 0.0
        N[a] += 1
        W[a] += v
    return N


def win_rate(noise, n_trials=200, n_sims=8, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([int(root_visits(n_sims, noise, rng).argmax()) == WIN for _ in range(n_trials)]))


print(win_rate(False), win_rate(True))
