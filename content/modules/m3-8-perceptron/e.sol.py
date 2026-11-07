import math
import torch
import torch.nn as nn


def _data(fn, n, seed):
    g = torch.Generator().manual_seed(seed)
    if fn == "f1":
        X = torch.rand(n, 1, generator=g) * 2 - 1
        return X, torch.sin(3 * math.pi * X) / (1 + X ** 2)
    X = torch.rand(n, 2, generator=g) * 2 - 1
    return X, torch.exp(-(X ** 2).sum(1, keepdim=True) / 0.1)


class BatchedMLP(nn.Module):
    def __init__(self, E, d_in, h, depth, d_out):
        super().__init__()
        dims = [d_in] + [h] * depth + [d_out]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for a, c in zip(dims[:-1], dims[1:]):
            bound = 1 / math.sqrt(a)
            self.W.append(nn.Parameter(torch.empty(E, a, c).uniform_(-bound, bound)))
            self.b.append(nn.Parameter(torch.empty(E, 1, c).uniform_(-bound, bound)))

    def forward(self, X):
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            X = torch.baddbmm(b, X, W)
            if i < len(self.W) - 1:
                X = torch.relu(X)
        return X


def train_batched(E, fn, depth, steps=1500, n=1000):
    data = [_data(fn, n, s) for s in range(E)]
    X = torch.stack([d[0] for d in data])
    Y = torch.stack([d[1] for d in data])
    Xt, Yt = _data(fn, 2000, 999)
    torch.manual_seed(0)
    m = BatchedMLP(E, X.shape[2], 64, depth, Y.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        loss = ((m(X) - Y) ** 2).mean(dim=(1, 2)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return ((m(Xt.expand(E, -1, -1)) - Yt) ** 2).mean(dim=(1, 2))


def train_loop(E, fn, depth, steps=1500, n=1000):
    out = []
    Xt, Yt = _data(fn, 2000, 999)
    for s in range(E):
        X, Y = _data(fn, n, s)
        torch.manual_seed(s)
        layers, d = [], X.shape[1]
        for _ in range(depth):
            layers += [nn.Linear(d, 64), nn.ReLU()]
            d = 64
        m = nn.Sequential(*layers, nn.Linear(d, Y.shape[1]))
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(steps):
            loss = ((m(X) - Y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out.append(float(((m(Xt) - Yt) ** 2).mean()))
    return torch.tensor(out)
