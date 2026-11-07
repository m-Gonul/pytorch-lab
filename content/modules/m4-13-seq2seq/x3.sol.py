import numpy as np


def full_grad(h, W, t):
    z = W @ h
    p = np.exp(z - z.max())
    p /= p.sum()
    return W.T @ p - W[t]


def sampled_grad(h, W, t, S, logq, correct=True):
    idx = np.concatenate([[t], S])
    z = W[idx] @ h
    if correct:
        z = z - np.concatenate([[0.0], logq[S]])
    p = np.exp(z - z.max())
    p /= p.sum()
    return W[idx].T @ p - W[t]
