import torch
import torch.nn.functional as F
from torch import nn


def ce_from_logits(Z, y):
    """Ortalama cross-entropy; logsumexp + gather ile."""
    return (torch.logsumexp(Z, dim=1) - Z.gather(1, y[:, None]).squeeze(1)).mean()


def train_softmax(X, y, K, steps=300, lr=0.5):
    model = nn.Linear(X.shape[1], K)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        loss = loss_fn(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


Z = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 10000.0]])
y = torch.tensor([0, 2])
print(ce_from_logits(Z, y), F.cross_entropy(Z, y))
