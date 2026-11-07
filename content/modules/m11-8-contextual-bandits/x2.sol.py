import numpy as np


def linear_bandit(T, policy, diverse, seed=0):
    rng = np.random.default_rng(seed)
    K, d, sigma = 2, 3, 0.5
    theta = rng.normal(0, 1, (K, d))
    P = np.repeat(np.eye(d)[None], K, 0)
    b = np.zeros((K, d))
    reg = 0.0
    for _ in range(T):
        x = rng.normal(0, 1, d) if diverse else np.r_[1.0, 0.05 * rng.normal(0, 1, d - 1)]
        S = np.linalg.inv(P)
        w = np.einsum("kij,kj->ki", S, b)
        if policy == "ts":
            w = w + np.einsum("kij,kj->ki", np.linalg.cholesky(S), rng.normal(0, 1, (K, d)))
        a = int((w @ x).argmax())
        m = theta @ x
        r = m[a] + sigma * rng.normal()
        P[a] += np.outer(x, x) / sigma ** 2
        b[a] += x * r / sigma ** 2
        reg += m.max() - m[a]
    return reg


print(linear_bandit(500, "greedy", True))
