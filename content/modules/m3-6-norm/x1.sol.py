import numpy as np


def bn_train(X, eps=1e-5):
    return (X - X.mean(0)) / np.sqrt(X.var(0) + eps)


def bn_noise(x, pool, B, trials, rng):
    outs = []
    for _ in range(trials):
        mates = pool[rng.choice(len(pool), B - 1, replace=False)]
        outs.append(bn_train(np.vstack([x[None], mates]))[0])
    return float(np.std(outs, 0).mean())


def running_stats(batches, momentum=0.1):
    m, v = np.zeros(batches[0].shape[1]), np.ones(batches[0].shape[1])
    for X in batches:
        n = len(X)
        m = (1 - momentum) * m + momentum * X.mean(0)
        v = (1 - momentum) * v + momentum * X.var(0) * n / (n - 1)
    return m, v
