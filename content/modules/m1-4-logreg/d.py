import torch
from torch import nn


def bce_logits(z, y):
    """Kararlı ikili cross-entropy (ortalama), doğrudan logit'ten."""
    ...


def train_torch(X, y, lr=0.5, steps=500):
    """→ eğitilmiş nn.Linear(d, 1)"""
    ...


z = torch.tensor([-1.0, 0.0, 2.0, -100.0, 100.0])
y = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
print(bce_logits(z, y))
