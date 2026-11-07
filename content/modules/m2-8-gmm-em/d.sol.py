import torch
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily


def make_scale_tril(raw):
    return torch.tril(raw, -1) + torch.diag_embed(torch.diagonal(raw, dim1=-2, dim2=-1).exp())


def gmm_mean_ll(X, logits, mus, raw):
    gmm = MixtureSameFamily(Categorical(logits=logits), MultivariateNormal(mus, scale_tril=make_scale_tril(raw)))
    return gmm.log_prob(X).mean()


def fit_gmm_gd(X, mus0, steps=800, lr=0.05):
    """→ son ortalama log-likelihood (float)."""
    K, d = mus0.shape
    logits = torch.zeros(K, dtype=X.dtype, requires_grad=True)
    mus = mus0.clone().requires_grad_(True)
    raw = torch.zeros(K, d, d, dtype=X.dtype, requires_grad=True)
    opt = torch.optim.Adam([logits, mus, raw], lr=lr)
    for _ in range(steps):
        loss = -gmm_mean_ll(X, logits, mus, raw)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(gmm_mean_ll(X, logits, mus, raw))
