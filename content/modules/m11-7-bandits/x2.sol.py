import numpy as np


def nonstationary(n_runs, n_steps, alpha=None, seed=0):
    rng = np.random.default_rng(seed)
    K = 10
    qs = np.zeros((n_runs, K))
    Q = np.zeros((n_runs, K))
    N = np.zeros((n_runs, K))
    ar = np.arange(n_runs)
    opt = np.empty(n_steps)
    for t in range(n_steps):
        greedy = (Q + 1e-9 * rng.random(Q.shape)).argmax(1)
        a = np.where(rng.random(n_runs) < 0.1, rng.integers(0, K, n_runs), greedy)
        opt[t] = (a == qs.argmax(1)).mean()
        r = rng.normal(qs[ar, a], 1)
        N[ar, a] += 1
        Q[ar, a] += (1 / N[ar, a] if alpha is None else alpha) * (r - Q[ar, a])
        qs += rng.normal(0, 0.01, qs.shape)
    return opt


print(nonstationary(200, 1000).mean())
