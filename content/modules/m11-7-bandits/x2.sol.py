import numpy as np


def nonstationary(n_runs, n_steps, alpha=None, seed=0, return_reward=False):
    rng = np.random.default_rng(seed)
    K = 10
    qs = np.zeros((n_runs, K))
    Q = np.zeros((n_runs, K))
    N = np.zeros((n_runs, K))
    ar = np.arange(n_runs)
    opt = np.empty(n_steps)
    rewards = np.empty(n_steps)
    for t in range(n_steps):
        greedy = (Q + 1e-9 * rng.random(Q.shape)).argmax(1)
        a = np.where(rng.random(n_runs) < 0.1, rng.integers(0, K, n_runs), greedy)
        opt[t] = (qs[ar, a] == qs.max(axis=1)).mean()
        r = rng.normal(qs[ar, a], 1)
        rewards[t] = r.mean()
        N[ar, a] += 1
        Q[ar, a] += (1 / N[ar, a] if alpha is None else alpha) * (r - Q[ar, a])
        qs += rng.normal(0, 0.01, qs.shape)
    return {"optimal": opt, "reward": rewards} if return_reward else opt


print(nonstationary(200, 1000).mean())
