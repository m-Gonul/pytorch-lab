import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def digits():
    X, y = load_digits(return_X_y=True)
    return torch.tensor(X, dtype=torch.float32) / 16, torch.tensor(y)


class AE(nn.Module):
    def __init__(self, sizes=(64, 32, 12), tied=False):
        super().__init__()
        ...

    def encode(self, x):
        ...

    def decode(self, z):
        ...

    def forward(self, x):
        ...


def train_ae(model, X, epochs=100, lr=3e-3, mask=0.0):
    ...


def greedy_pretrain(sizes, X, epochs):
    ...


def classify(encoder_ae, X, y, Xt, yt, freeze, steps=300):
    ...
