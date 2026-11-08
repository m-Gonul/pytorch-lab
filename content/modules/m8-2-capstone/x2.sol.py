import numpy as np

MU = np.array([[-0.6, -2.0], [0.6, 2.0]])
SIG = 0.3
BETAS = np.linspace(1e-4, 0.05, 200)
AB = np.cumprod(1 - BETAS)


def eps(x, ab):
    v = ab * SIG ** 2 + 1 - ab
    m = np.sqrt(ab) * MU
    lw = -((x[:, None] - m[None]) ** 2).sum(-1) / (2 * v)
    r = np.exp(lw - lw.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    return np.sqrt(1 - ab) * (x - r @ m) / v


def repaint(xk, n, rng, U):
    x = rng.normal(size=(n, 2))
    for t in range(len(BETAS) - 1, -1, -1):
        abp = AB[t - 1] if t > 0 else 1.0
        for u in range(U):
            e = eps(x, AB[t])
            x0 = (x - np.sqrt(1 - AB[t]) * e) / np.sqrt(AB[t])
            mean = np.sqrt(abp) * BETAS[t] / (1 - AB[t]) * x0 + np.sqrt(1 - BETAS[t]) * (1 - abp) / (1 - AB[t]) * x
            var = (1 - abp) / (1 - AB[t]) * BETAS[t]
            xu = mean + (np.sqrt(var) * rng.normal(size=x.shape) if t > 0 else 0.0)
            xu[:, 0] = np.sqrt(abp) * xk + (np.sqrt(1 - abp) * rng.normal(size=n) if t > 0 else 0.0)
            if u < U - 1 and t > 0:
                x = np.sqrt(1 - BETAS[t]) * xu + np.sqrt(BETAS[t]) * rng.normal(size=x.shape)
            else:
                x = xu
    return x
