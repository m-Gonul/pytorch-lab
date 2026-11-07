import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def momentum_path_torch(w0, lr, beta, steps):
    """L(w) = (w − 3)² için SGD(momentum=beta) yörüngesi (liste)."""
    ...


def train_torch(X, y, lr=0.05, batch_size=32, epochs=50, momentum=0.9):
    ...


print(momentum_path_torch(0.0, 0.25, 0.5, 4))    # [1.5, 3.0, …]
