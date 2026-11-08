import numpy as np


def inception_score(P):
    py = P.mean(0, keepdims=True)
    kl = np.sum(P * (np.log(np.clip(P, 1e-12, None)) - np.log(np.clip(py, 1e-12, None))), 1)
    return float(np.exp(kl.mean()))


def fid(F1, F2):
    m1, m2 = F1.mean(0), F2.mean(0)
    S1, S2 = np.cov(F1, rowvar=False), np.cov(F2, rowvar=False)
    w, V = np.linalg.eigh(S1)
    r = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    tr = np.sqrt(np.clip(np.linalg.eigvalsh(r @ S2 @ r), 0, None)).sum()
    return float(np.sum((m1 - m2) ** 2) + np.trace(S1) + np.trace(S2) - 2 * tr)
