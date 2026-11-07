import torch


def unbroadcast_torch(G, shape):
    ...


def manual_grads(a, b, W):
    """L = ReLU((a * b + a) @ W).sum() için (da, db, dW) — autograd YOK."""
    ...
