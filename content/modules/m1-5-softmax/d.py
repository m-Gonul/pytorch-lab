import torch
import torch.nn.functional as F
from torch import nn


def ce_from_logits(Z, y):
    """Ortalama cross-entropy; logsumexp + gather ile."""
    ...


def train_softmax(X, y, K, steps=300, lr=0.5):
    ...


Z = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 10000.0]])
y = torch.tensor([0, 2])
print(ce_from_logits(Z, y), F.cross_entropy(Z, y))
