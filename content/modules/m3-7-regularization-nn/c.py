import numpy as np


def dropout_np(X, p, rng):
    ...


def mc_predict(W1, b1, W2, b2, X, p, T, rng):
    """→ (mean, std), (N, C). Döngü yok."""
    ...


class EarlyStopper:
    def __init__(self, patience, min_delta=0.0):
        ...

    def step(self, val_loss):
        ...
