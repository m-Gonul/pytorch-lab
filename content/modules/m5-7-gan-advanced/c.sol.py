import numpy as np


def sqrtm_psd(A):
    w, V = np.linalg.eigh((A + A.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def sqrtm_ns(A, iters=50):
    n = np.linalg.norm(A)
    Y, Z = A / n, np.eye(len(A))
    for _ in range(iters):
        T = 0.5 * (3 * np.eye(len(A)) - Z @ Y)
        Y, Z = Y @ T, T @ Z
    return Y * np.sqrt(n)


def frechet(mu1, S1, mu2, S2):
    r = sqrtm_psd(S1)
    tr = np.sqrt(np.clip(np.linalg.eigvalsh(r @ S2 @ r), 0, None)).sum()
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(S1) + np.trace(S2) - 2 * tr)


def fid(F1, F2):
    return frechet(F1.mean(0), np.cov(F1, rowvar=False), F2.mean(0), np.cov(F2, rowvar=False))


def inception_score(P, splits=1):
    parts = np.array_split(P, splits)
    s = []
    for q in parts:
        py = q.mean(0, keepdims=True)
        kl = np.sum(q * (np.log(np.clip(q, 1e-12, None)) - np.log(np.clip(py, 1e-12, None))), 1)
        s.append(np.exp(kl.mean()))
    return float(np.mean(s)), float(np.std(s))
