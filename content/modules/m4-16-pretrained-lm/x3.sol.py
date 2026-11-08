import numpy as np


def electra_corrupt(X, rng, gen_probs, n_special, p=0.15):
    sel = (rng.random(X.shape) < p) & (X >= n_special)
    c = gen_probs.cumsum(-1)
    u = rng.random(X.shape)[..., None]
    samp = (u > c).sum(-1)
    samp = np.minimum(samp, gen_probs.shape[-1] - 1)
    Xc = np.where(sel, samp, X)
    return Xc, Xc != X
