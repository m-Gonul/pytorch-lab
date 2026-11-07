import numpy as np


def symmetry_perms():
    g = np.arange(9).reshape(3, 3)
    out = []
    for k in range(4):
        r = np.rot90(g, k)
        out += [r.ravel(), np.fliplr(r).ravel()]
    return np.array(out)


def augment(X, PI, Z):
    P = symmetry_perms()
    Xa = np.concatenate([X[:, :9][:, P], X[:, 9:][:, P]], 2)
    return Xa.transpose(1, 0, 2).reshape(-1, 18), PI[:, P].transpose(1, 0, 2).reshape(-1, 9), np.tile(Z, 8)


def az_loss(z, v, pi, logits, theta_sq, c=1e-4):
    m = logits.max(1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(1, keepdims=True))
    return float(np.mean((z - v) ** 2) - np.mean((pi * logp).sum(1)) + c * theta_sq)


def elo_expected(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def fit_elo(Wm, n_iter=500):
    Wm = Wm - np.diag(np.diag(Wm))
    n = Wm + Wm.T
    W = Wm.sum(1)
    g = np.ones(len(W))
    for _ in range(n_iter):
        d = n / (g[:, None] + g[None, :])
        np.fill_diagonal(d, 0)
        g = W / d.sum(1)
        g = g / np.exp(np.log(g).mean())
    r = 400 * np.log10(g)
    return r - r.mean()


print(symmetry_perms().shape)
