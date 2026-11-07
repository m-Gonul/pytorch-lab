import copy
import torch
import torch.nn.functional as F
from torch import nn


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        ...


def fit_early_stopping(model, Xtr, ytr, Xva, yva, patience=10, max_epochs=500, lr=1e-2):
    """→ (model [en iyi ağırlıklar yüklü], best_epoch, history)"""
    ...
