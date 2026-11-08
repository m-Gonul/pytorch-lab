import numpy as np


def fid(F1, F2):
    m1, m2 = F1.mean(0), F2.mean(0)
    S1, S2 = np.cov(F1, rowvar=False), np.cov(F2, rowvar=False)
    w, V = np.linalg.eigh(S1)
    r = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    tr = np.sqrt(np.clip(np.linalg.eigvalsh(r @ S2 @ r), 0, None)).sum()
    return float(np.sum((m1 - m2) ** 2) + np.trace(S1) + np.trace(S2) - 2 * tr)


def fid_curve(F_real, sampler, Ns, seed):
    rng = np.random.default_rng(seed)
    return np.array([fid(F_real, sampler(n, rng)) for n in Ns])


def fid_inf(Ns, fids):
    A = np.stack([np.ones(len(Ns)), 1.0 / np.asarray(Ns, float)], 1)
    return float(np.linalg.lstsq(A, np.asarray(fids), rcond=None)[0][0])
