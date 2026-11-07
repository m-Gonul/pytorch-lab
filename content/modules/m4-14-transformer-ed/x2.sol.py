import numpy as np


def _attn(q, K, V, mask=None):
    s = q @ K.T / np.sqrt(K.shape[1])
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s)
    return (a / a.sum(-1, keepdims=True)) @ V


def decoder_full(X, Wq, Wk, Wv, causal=True):
    T = len(X)
    return _attn(X @ Wq, X @ Wk, X @ Wv, np.tril(np.ones((T, T), bool)) if causal else None)


def decoder_incremental(X, Wq, Wk, Wv):
    return np.stack([_attn(X[t:t + 1] @ Wq, X[:t + 1] @ Wk, X[:t + 1] @ Wv)[0] for t in range(len(X))])
