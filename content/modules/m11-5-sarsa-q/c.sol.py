import numpy as np


def eps_greedy_probs(Q, eps):
    n, A = Q.shape
    p = np.full((n, A), eps / A)
    p[np.arange(n), Q.argmax(1)] += 1 - eps
    return p


def _choose(rng, Q, eps):
    n, A = Q.shape
    greedy = (Q + 1e-9 * rng.random(Q.shape)).argmax(1)
    return np.where(rng.random(n) < eps, rng.integers(0, A, n), greedy)


def max_bias(n_runs, n_eps, alpha=0.1, eps=0.1, n_b=10, seed=0, double=False):
    rng = np.random.default_rng(seed)
    ar = np.arange(n_runs)
    QA = np.zeros((2, n_runs, 2))
    QB = np.zeros((2, n_runs, n_b))
    left = np.empty(n_eps)
    for e in range(n_eps):
        k = rng.integers(0, 2, n_runs) if double else np.zeros(n_runs, int)
        o = 1 - k if double else k
        a = _choose(rng, QA[0] + QA[1] if double else QA[0], eps)
        go = a == 0
        left[e] = go.mean()
        vB = QB[o, ar, QB[k, ar].argmax(1)]
        QA[k, ar, a] += alpha * (np.where(go, vB, 0.0) - QA[k, ar, a])
        b = _choose(rng, QB[0] + QB[1] if double else QB[0], eps)
        r = rng.normal(-0.1, 1.0, n_runs)
        QB[k, ar, b] += np.where(go, alpha * (r - QB[k, ar, b]), 0.0)
    return left


print(max_bias(1000, 100)[[0, 10, 99]])
