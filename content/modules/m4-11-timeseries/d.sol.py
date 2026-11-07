import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_time_series(n, steps, seed):
    g = torch.Generator().manual_seed(seed)
    f1, f2, o1, o2 = torch.rand(4, n, 1, generator=g)
    t = torch.linspace(0, 1, steps)
    s = 0.5 * torch.sin((t - o1) * (f1 * 10 + 10)) + 0.2 * torch.sin((t - o2) * (f2 * 20 + 20))
    s = s + 0.1 * (torch.rand(n, steps, generator=g) - 0.5)
    return s[..., None]


class RNNStack(nn.Module):
    def __init__(self, sizes, head=None, seq=False):
        super().__init__()
        self.rnns = nn.ModuleList([nn.RNN(a, b, batch_first=True) for a, b in zip([1] + sizes[:-1], sizes)])
        self.head, self.seq = head, seq

    def forward(self, x):
        for r in self.rnns:
            x, _ = r(x)
        if self.head is None:
            return x[:, -1]
        return self.head(x) if self.seq else self.head(x[:, -1])


def ex6_models():
    return {1: nn.Sequential(nn.Flatten(), nn.Linear(40, 1)), 2: RNNStack([1]), 3: RNNStack([16, 16, 1]),
            4: RNNStack([16, 16], nn.Linear(16, 8)), 5: RNNStack([16, 16], nn.Linear(16, 8), seq=True)}


def seq_targets(S, n_steps=40, ahead=8):
    return torch.stack([S[:, k:k + n_steps, 0] for k in range(1, ahead + 1)], -1)


def train(m, X, Y, epochs=10, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            idx = perm[i:i + 32]
            loss = F.mse_loss(m(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


def iterative_forecast(m, X, h, window=40):
    with torch.no_grad():
        for _ in range(h):
            X = torch.cat([X, m(X[:, -window:])[:, None]], 1)
    return X[:, -h:, 0]
