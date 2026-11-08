import numpy as np


def paired_bootstrap(a, b, B, rng, alpha=0.05):
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    lo, hi = np.quantile(m, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def unpaired_bootstrap(a, b, B, rng, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    mb = b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    lo, hi = np.quantile(ma - mb, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)
