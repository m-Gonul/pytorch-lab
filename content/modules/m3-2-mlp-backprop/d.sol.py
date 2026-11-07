import torch
import torch.nn.functional as F
from torch import nn


def two_layer_torch(X, y, W1, b1, W2, b2, reg):
    S = torch.relu(X @ W1 + b1) @ W2 + b2
    return F.cross_entropy(S, y) + reg * ((W1 ** 2).sum() + (W2 ** 2).sum())


def train_mlp(X, y, H, C, epochs=300, lr=0.05):
    model = nn.Sequential(nn.Linear(X.shape[1], H), nn.ReLU(), nn.Linear(H, C))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
