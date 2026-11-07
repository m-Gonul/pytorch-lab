import torch


def mc_logits(W1, b1, W2, b2, X, masks, p):
    """masks: (T, N, H) → (T, N, C). Döngü yok."""
    H = torch.relu(X @ W1 + b1)                       # (N, H)
    return (H * masks.to(H.dtype) / (1 - p)) @ W2 + b2


def predictive_entropy(logits):
    """(T, N, C) → (N,). Döngü yok."""
    P = logits.softmax(-1).mean(0)
    return -(P * P.clamp_min(1e-12).log()).sum(-1)
