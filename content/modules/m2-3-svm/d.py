import torch
import torch.nn.functional as F
from torch import nn


def svm_loss_torch(S, y, delta=1.0):
    """S: (n, C) skorlar → ortalama çok sınıflı hinge (CS231n). Elle."""
    ...


def train_linear_svm(X, y, C, steps=500, lr=0.1, wd=1e-3):
    ...


S = torch.tensor([[3.2, 5.1, -1.7]])
print(svm_loss_torch(S, torch.tensor([0])))   # 2.9
