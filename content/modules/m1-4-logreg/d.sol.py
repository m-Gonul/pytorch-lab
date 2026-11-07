import torch
from torch import nn


def bce_logits(z, y):
    """Kararlı ikili cross-entropy (ortalama), doğrudan logit'ten."""
    # softplus(z) − z·y; logaddexp hem taşmaz hem her yerde pürüzsüz (doğru gradyan)
    return (torch.logaddexp(torch.zeros_like(z), z) - z * y).mean()


def train_torch(X, y, lr=0.5, steps=500):
    """→ eğitilmiş nn.Linear(d, 1)"""
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    t = y.float().reshape(-1, 1)
    for _ in range(steps):
        loss = loss_fn(model(X), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


z = torch.tensor([-1.0, 0.0, 2.0, -100.0, 100.0])
y = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
print(bce_logits(z, y))
