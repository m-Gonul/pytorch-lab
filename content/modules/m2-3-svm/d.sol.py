import torch
import torch.nn.functional as F
from torch import nn


def svm_loss_torch(S, y, delta=1.0):
    """S: (n, C) skorlar → ortalama çok sınıflı hinge (CS231n). Elle."""
    sy = S.gather(1, y[:, None])
    M = (S - sy + delta).clamp(min=0)
    M = M.scatter(1, y[:, None], 0.0)
    return M.sum(1).mean()


def train_linear_svm(X, y, C, steps=500, lr=0.1, wd=1e-3):
    model = nn.Linear(X.shape[1], C)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        loss = svm_loss_torch(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


S = torch.tensor([[3.2, 5.1, -1.7]])
print(svm_loss_torch(S, torch.tensor([0])))   # 2.9
