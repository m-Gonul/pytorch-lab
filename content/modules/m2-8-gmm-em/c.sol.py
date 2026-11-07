import numpy as np


def log_gauss(X, mus, covs):
    """→ (n, K). Döngü yok."""
    d = X.shape[1]
    diff = X[:, None, :] - mus[None]                              # (n, K, d)
    P = np.linalg.inv(covs)                                       # (K, d, d)
    maha = np.einsum("nkd,kde,nke->nk", diff, P, diff)
    logdet = np.linalg.slogdet(covs)[1]                           # (K,)
    return -0.5 * (d * np.log(2 * np.pi) + logdet[None] + maha)


def e_step_np(X, pis, mus, covs):
    L = np.log(pis)[None] + log_gauss(X, mus, covs)
    m = L.max(1, keepdims=True)
    lse = m + np.log(np.exp(L - m).sum(1, keepdims=True))
    return np.exp(L - lse), float(lse.sum())


def m_step_np(X, R, reg=1e-6):
    Nk = R.sum(0)
    mus = R.T @ X / Nk[:, None]
    diff = X[:, None, :] - mus[None]
    covs = np.einsum("nk,nkd,nke->kde", R, diff, diff) / Nk[:, None, None] + reg * np.eye(X.shape[1])
    return Nk / len(X), mus, covs


def em_np(X, pis, mus, covs, iters):
    lls = []
    for _ in range(iters):
        R, ll = e_step_np(X, pis, mus, covs)
        lls.append(ll)
        pis, mus, covs = m_step_np(X, R)
    lls.append(e_step_np(X, pis, mus, covs)[1])
    return pis, mus, covs, np.array(lls)


rng = np.random.default_rng(0)
X = np.concatenate([rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 150), rng.normal([4, 0], 0.5, (100, 2))])
pis, mus, covs, lls = em_np(X, np.array([0.5, 0.5]), X[[0, -1]], np.stack([np.eye(2)] * 2), 30)
print(mus, lls[-1])
