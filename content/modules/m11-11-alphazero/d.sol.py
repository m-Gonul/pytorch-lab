import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]


def term(b):
    for l in LINES:
        s = b[l[0]] + b[l[1]] + b[l[2]]
        if s == 3 or s == -3:
            return s // 3
    return 0 if all(b) else None


class Net(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(18, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.p = nn.Linear(h, 9)
        self.v = nn.Linear(h, 1)

    def forward(self, x):
        z = self.body(x)
        return self.p(z), torch.tanh(self.v(z)).squeeze(-1)


def encode(board, player):
    b = torch.tensor(board, dtype=torch.float32) * player
    return torch.cat([(b == 1).float(), (b == -1).float()])


class _Node:
    def __init__(self, board, player, net):
        self.b, self.pl = board, player
        self.legal = np.array([v == 0 for v in board])
        with torch.no_grad():
            lo, v = net(encode(board, player)[None])
        p = torch.softmax(lo[0], 0).numpy() * self.legal
        self.P, self.v = p / p.sum(), float(v[0])
        self.N, self.W, self.ch = np.zeros(9), np.zeros(9), {}


def mcts(net, board, player, n_sims, rng, c=1.5, noise=True):
    root = _Node(list(board), player, net)
    if noise:
        eta = np.zeros(9)
        eta[root.legal] = rng.dirichlet([0.3] * int(root.legal.sum()))
        root.P = 0.75 * root.P + 0.25 * eta
    for _ in range(n_sims):
        node, path = root, []
        while True:
            Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
            U = c * node.P * math.sqrt(node.N.sum() + 1) / (1 + node.N)
            a = int(np.argmax(np.where(node.legal, Q + U, -1e9)))
            path.append((node, a))
            if a in node.ch:
                node = node.ch[a]
                continue
            b2 = node.b[:]
            b2[a] = node.pl
            t = term(b2)
            if t is not None:
                v = -abs(t)
            else:
                node.ch[a] = _Node(b2, -node.pl, net)
                v = node.ch[a].v
            break
        for nd, a in reversed(path):
            v = -v
            nd.N[a] += 1
            nd.W[a] += v
    return root.N


def self_play(net, n_sims, rng):
    b, pl, hist = [0] * 9, 1, []
    while term(b) is None:
        N = mcts(net, b, pl, n_sims, rng)
        pi = N / N.sum()
        a = int(rng.choice(9, p=pi)) if sum(1 for v in b if v) < 3 else int(N.argmax())
        hist.append((b[:], pl, pi))
        b[a] = pl
        pl = -pl
    z = term(b)
    return [(bb, p, pi, z * p) for bb, p, pi in hist]


def _perms():
    g = np.arange(9).reshape(3, 3)
    return [np.ascontiguousarray(f(np.rot90(g, k))).ravel() for k in range(4) for f in (lambda x: x, np.fliplr)]


def train(net, opt, data, epochs=3, bs=64):
    X = torch.stack([encode(b, p) for b, p, _, _ in data])
    PI = torch.tensor(np.array([pi for _, _, pi, _ in data]), dtype=torch.float32)
    Z = torch.tensor([z for *_, z in data], dtype=torch.float32)
    idx = [torch.tensor(p) for p in _perms()]
    X = torch.cat([torch.cat([X[:, :9][:, i], X[:, 9:][:, i]], 1) for i in idx])
    PI = torch.cat([PI[:, i] for i in idx])
    Z = Z.repeat(8)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            lo, v = net(X[j])
            loss = F.mse_loss(v, Z[j]) - (PI[j] * F.log_softmax(lo, 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss.detach())


def alphazero(iters=15, games=20, n_sims=30, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    buf = []
    for _ in range(iters):
        for _ in range(games):
            buf += self_play(net, n_sims, rng)
        buf = buf[-3000:]
        train(net, opt, buf)
    return net


net = alphazero(2, 5, 10)
print(mcts(net, [0] * 9, 1, 30, np.random.default_rng(0), noise=False))
