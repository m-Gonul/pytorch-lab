import numpy as np


def fit_pca(X, k):
    m = X.mean(0)
    _, _, Vt = np.linalg.svd(X - m, full_matrices=False)
    return m, Vt[:k]


def recon_error(X, mean, comps):
    Z = (X - mean) @ comps.T
    return np.sum((Z @ comps + mean - X) ** 2, 1)


def roc_auc(scores, labels):
    s = np.asarray(scores, float)
    lab = np.asarray(labels).astype(bool)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    npos, nneg = lab.sum(), (~lab).sum()
    return float((ranks[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg))
