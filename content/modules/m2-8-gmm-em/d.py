import torch
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily


def make_scale_tril(raw):
    ...


def gmm_mean_ll(X, logits, mus, raw):
    ...


def fit_gmm_gd(X, mus0, steps=800, lr=0.05):
    """→ son ortalama log-likelihood (float)."""
    ...
