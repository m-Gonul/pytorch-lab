import numpy as np


def testbed(n_runs, n_steps, eps, Q0=0.0, alpha=None, seed=0):
    rng = np.random.default_rng(seed)
    K = 10
    qs = rng.normal(0, 1, (n_runs, K))
    best = qs.argmax(1)
    Q = np.full((n_runs, K), float(Q0))
    N = np.zeros((n_runs, K))
    ar = np.arange(n_runs)
    rew, opt = np.empty(n_steps), np.empty(n_steps)
    for t in range(n_steps):
        greedy = (Q + 1e-9 * rng.random(Q.shape)).argmax(1)
        a = np.where(rng.random(n_runs) < eps, rng.integers(0, K, n_runs), greedy)
        r = rng.normal(qs[ar, a], 1)
        N[ar, a] += 1
        step = 1 / N[ar, a] if alpha is None else alpha
        Q[ar, a] += step * (r - Q[ar, a])
        rew[t], opt[t] = r.mean(), (a == best).mean()
    return rew, opt


def gradient_bandit(n_runs, n_steps, alpha, baseline=True, seed=0, offset=4.0):
    rng = np.random.default_rng(seed)
    K = 10
    qs = rng.normal(offset, 1, (n_runs, K))
    best = qs.argmax(1)
    H = np.zeros((n_runs, K))
    rbar = np.zeros(n_runs)
    ar = np.arange(n_runs)
    rew, opt = np.empty(n_steps), np.empty(n_steps)
    for t in range(n_steps):
        e = np.exp(H - H.max(1, keepdims=True))
        pi = e / e.sum(1, keepdims=True)
        a = (rng.random((n_runs, 1)) > pi.cumsum(1)).sum(1).clip(max=K - 1)
        r = rng.normal(qs[ar, a], 1)
        if baseline:
            rbar += (r - rbar) / (t + 1)
        one = np.zeros_like(H)
        one[ar, a] = 1
        H += alpha * (r - (rbar if baseline else 0.0))[:, None] * (one - pi)
        rew[t], opt[t] = r.mean(), (a == best).mean()
    return rew, opt


r, opt = testbed(200, 500, 0.1)
print(r[-50:].mean(), opt[-50:].mean())
