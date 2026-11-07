import torch
import torch.nn.functional as F


def fit_reg_stump(X, r):
    """(Hazır) X: (n, d), r: (n,) → (j, t, sol, sağ). SSE'yi en çok azaltan stump."""
    n, d = X.shape
    Xs, order = torch.sort(X, dim=0, stable=True)
    Rs = r[order]
    SL = Rs.cumsum(0)[:-1]
    SR = Rs.sum(0) - SL
    nl = torch.arange(1, n, dtype=X.dtype)[:, None]
    gain = torch.where(Xs[:-1] == Xs[1:], -torch.inf, SL ** 2 / nl + SR ** 2 / (n - nl))
    k = int(gain.T.reshape(-1).argmax()); j, i = divmod(k, n - 1)
    return j, float((Xs[i, j] + Xs[i + 1, j]) / 2), float(SL[i, j] / (i + 1)), float(SR[i, j] / (n - i - 1))


def stump(st, X):
    j, t, l, r = st
    return torch.where(X[:, j] <= t, l, r).to(X.dtype)


def pseudo_residuals(F, y, loss_fn):
    """−∂ Σ loss_fn(F, y) / ∂F, autograd ile."""
    ...


def gbm_classify(X, y, rounds=200, lr=0.3):
    ...


def gbm_logits(model, X):
    ...
