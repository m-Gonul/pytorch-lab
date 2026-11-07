import torch


def mc_logits(W1, b1, W2, b2, X, masks, p):
    """masks: (T, N, H) → (T, N, C). Döngü yok."""
    ...


def predictive_entropy(logits):
    """(T, N, C) → (N,). Döngü yok."""
    ...
