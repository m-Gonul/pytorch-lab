import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def momentum_path_torch(w0, lr, beta, steps):
    """L(w) = (w − 3)² için SGD(momentum=beta) yörüngesi (liste)."""
    w = torch.tensor(float(w0), requires_grad=True)
    opt = torch.optim.SGD([w], lr=lr, momentum=beta)
    path = []
    for _ in range(steps):
        opt.zero_grad()
        ((w - 3) ** 2).backward()
        opt.step()
        path.append(w.item())
    return path


def train_torch(X, y, lr=0.05, batch_size=32, epochs=50, momentum=0.9):
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loader = DataLoader(TensorDataset(X, y.reshape(-1, 1)), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            loss = ((model(xb) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


print(momentum_path_torch(0.0, 0.25, 0.5, 4))    # [1.5, 3.0, …]
