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
    F = F.detach().requires_grad_(True)
    g, = torch.autograd.grad(loss_fn(F, y).sum(), F)
    return -g


def bce(F_, y):
    return F.binary_cross_entropy_with_logits(F_, y, reduction="none")


def gbm_classify(X, y, rounds=200, lr=0.3):
    p = y.mean()
    F0 = float(torch.log(p / (1 - p)))
    Fx = torch.full((X.shape[0],), F0, dtype=X.dtype)
    stumps = []
    for _ in range(rounds):
        r = pseudo_residuals(Fx, y, bce)
        st = fit_reg_stump(X, r)
        stumps.append(st)
        Fx = Fx + lr * stump(st, X)
    return F0, lr, stumps


def gbm_logits(model, X):
    F0, lr, stumps = model
    out = torch.full((X.shape[0],), F0, dtype=X.dtype)
    for st in stumps:
        out = out + lr * stump(st, X)
    return out
