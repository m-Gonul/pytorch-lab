import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Linear(c, c // r)
        self.fc2 = nn.Linear(c // r, c)

    def forward(self, x):
        s = x.mean((2, 3))
        return x * torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))[:, :, None, None]


class Inception(nn.Module):
    def __init__(self, cin, c1, c3r, c3, c5r, c5, pp):
        super().__init__()
        cbr = lambda a, b, k: nn.Sequential(nn.Conv2d(a, b, k, padding=k // 2), nn.ReLU())
        self.b1 = cbr(cin, c1, 1)
        self.b3 = nn.Sequential(cbr(cin, c3r, 1), cbr(c3r, c3, 3))
        self.b5 = nn.Sequential(cbr(cin, c5r, 1), cbr(c5r, c5, 5))
        self.bp = nn.Sequential(nn.MaxPool2d(3, 1, 1), cbr(cin, pp, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], 1)


def ex5_models():
    P = lambda: nn.MaxPool2d(2)
    c = lambda a, b, k: nn.Conv2d(a, b, k, padding=k // 2)
    fc1 = nn.Sequential(nn.Flatten(), nn.Linear(784, 300), nn.ReLU(), nn.Linear(300, 100), nn.ReLU(), nn.Linear(100, 10))
    cnn2 = nn.Sequential(c(1, 16, 7), nn.ReLU(), P(), c(16, 32, 3), nn.ReLU(), P(), c(32, 64, 3), nn.ReLU(), P(),
                         nn.Flatten(), nn.Linear(576, 64), nn.ReLU(), nn.Linear(64, 10))
    fc3 = nn.Sequential(nn.Flatten(), nn.Linear(784, 74), nn.ReLU(), nn.Linear(74, 50), nn.ReLU(), nn.Linear(50, 10))
    B = nn.BatchNorm2d
    cnn4 = nn.Sequential(c(1, 64, 7), nn.ReLU(), B(64), P(), B(64), c(64, 128, 5), nn.ReLU(), B(128), c(128, 128, 3), nn.ReLU(), B(128), P(), B(128),
                         c(128, 256, 3), nn.ReLU(), B(256), c(256, 256, 3), nn.ReLU(), B(256), P(),
                         nn.Flatten(), nn.Linear(2304, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    return {"fc1": fc1, "cnn2": cnn2, "fc3": fc3, "cnn4": cnn4}


def shifted_digits(n_rep, seed):
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 8, 8) / 16
    big = F.interpolate(X, size=(20, 20), mode="bilinear", align_corners=False).repeat(n_rep, 1, 1, 1)
    g = torch.Generator().manual_seed(seed)
    n = len(big)
    oy = torch.randint(0, 9, (n,), generator=g)
    ox = torch.randint(0, 9, (n,), generator=g)
    rows = (oy[:, None] + torch.arange(20))[:, :, None].expand(n, 20, 20)
    cols = (ox[:, None] + torch.arange(20))[:, None, :].expand(n, 20, 20)
    out = torch.zeros(n, 1, 28, 28)
    out[torch.arange(n)[:, None, None], 0, rows, cols] = big[:, 0]
    return out, torch.tensor(y).repeat(n_rep)


def train_eval(model, X, y, Xt, yt, epochs=8):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        p = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = p[i:i + 64]
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return float((model(Xt).argmax(1) == yt).float().mean())
