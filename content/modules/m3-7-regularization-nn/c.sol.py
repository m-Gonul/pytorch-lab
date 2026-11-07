import numpy as np


def dropout_np(X, p, rng):
    mask = rng.random(X.shape) >= p
    return X * mask / (1 - p), mask


def mc_predict(W1, b1, W2, b2, X, p, T, rng):
    """→ (mean, std), (N, C). Döngü yok."""
    H = np.maximum(0, X @ W1 + b1)                                    # (N, H)
    masks = rng.random((T,) + H.shape) >= p                           # (T, N, H)
    S = (H * masks / (1 - p)) @ W2 + b2                               # (T, N, C)
    S = S - S.max(-1, keepdims=True)
    P = np.exp(S) / np.exp(S).sum(-1, keepdims=True)
    return P.mean(0), P.std(0)


class EarlyStopper:
    def __init__(self, patience, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.epoch, self.bad = np.inf, -1, -1, 0

    def step(self, val_loss):
        self.epoch += 1
        if val_loss < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad = val_loss, self.epoch, 0
            return False
        self.bad += 1
        return self.bad >= self.patience
