import torch
from torch import nn


class SoftStump(nn.Module):
    def __init__(self, d, C, tau=0.5):
        super().__init__()
        self.gate = nn.Linear(d, 1)
        self.leaf_left = nn.Parameter(torch.zeros(C))
        self.leaf_right = nn.Parameter(torch.zeros(C))
        self.tau = tau

    def forward(self, x):
        g = torch.sigmoid(self.gate(x) / self.tau)            # (n, 1)
        return g * self.leaf_right + (1 - g) * self.leaf_left


def train_soft_stump(X, y, C, steps=500, lr=0.1):
    model = SoftStump(X.shape[1], C)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        loss = loss_fn(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


torch.manual_seed(0)
X = torch.rand(300, 4)
y = (X[:, 2] > 0.6).long()
m = train_soft_stump(X, y, 2)
print((m(X).argmax(1) == y).float().mean())
