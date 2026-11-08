import numpy as np


def _s(z):
    return 1 / (1 + np.exp(-z))


def retention(bf, T):
    return _s(np.asarray(bf, float)) ** T


def chrono_bias(h, Tmax, rng):
    bf = np.log(rng.uniform(1, Tmax - 1, h))
    return bf, -bf


def memory_curve(bf, T, h=64, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (4 * h, 1))
    U = rng.normal(0, 0.1, (4 * h, h))
    b = np.zeros(4 * h)
    b[h:2 * h] = bf

    def run(x0):
        hh, c = np.zeros(h), np.zeros(h)
        cs = []
        for t in range(T + 1):
            x = np.array([x0 if t == 0 else 0.0])
            a = W @ x + U @ hh + b
            i, f, g, o = _s(a[:h]), _s(a[h:2 * h]), np.tanh(a[2 * h:3 * h]), _s(a[3 * h:])
            c = f * c + i * g
            hh = o * np.tanh(c)
            cs.append(c.copy())
        return np.array(cs)
    eps = 1e-5
    d = (run(eps) - run(0.0)) / eps
    return np.linalg.norm(d[1:], axis=1)
