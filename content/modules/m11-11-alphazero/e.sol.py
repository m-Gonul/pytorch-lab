import math
import numpy as np
import torch
import torch.nn as nn

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


def encode_batch(boards, players):
    B = torch.tensor(boards, dtype=torch.float32) * torch.tensor(players, dtype=torch.float32)[:, None]
    return torch.cat([(B == 1).float(), (B == -1).float()], 1)


class _Node:
    __slots__ = ("b", "pl", "legal", "P", "v", "N", "W", "ch")

    def __init__(self, b, pl):
        self.b, self.pl = b, pl
        self.legal = np.array([x == 0 for x in b])
        self.P, self.v = None, 0.0
        self.N, self.W, self.ch = np.zeros(9), np.zeros(9), {}


def _evaluate(net, nodes):
    with torch.no_grad():
        lo, v = net(encode_batch([n.b for n in nodes], [n.pl for n in nodes]))
    pr = torch.softmax(lo, 1).numpy()
    for n, p, vv in zip(nodes, pr, v.numpy()):
        p = p * n.legal
        n.P, n.v = p / p.sum(), float(vv)


def batched_self_play(net, n_games, n_sims, rng, c=1.5):
    boards = [[0] * 9 for _ in range(n_games)]
    players = [1] * n_games
    hist = [[] for _ in range(n_games)]
    active, out = list(range(n_games)), []
    while active:
        roots = [_Node(boards[g][:], players[g]) for g in active]
        _evaluate(net, roots)
        for r in roots:
            eta = np.zeros(9)
            eta[r.legal] = rng.dirichlet([0.3] * int(r.legal.sum()))
            r.P = 0.75 * r.P + 0.25 * eta
        for _ in range(n_sims):
            leaves = []
            for r in roots:
                node, path = r, []
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
                        leaves.append((path, None, -abs(t)))
                    else:
                        ch = _Node(b2, -node.pl)
                        node.ch[a] = ch
                        leaves.append((path, ch, None))
                    break
            new = [ch for _, ch, _ in leaves if ch is not None]
            if new:
                _evaluate(net, new)
            for path, ch, tv in leaves:
                v = tv if ch is None else ch.v
                for nd, a in reversed(path):
                    v = -v
                    nd.N[a] += 1
                    nd.W[a] += v
        still = []
        for g, r in zip(active, roots):
            pi = r.N / r.N.sum()
            a = int(rng.choice(9, p=pi)) if sum(1 for x in boards[g] if x) < 3 else int(r.N.argmax())
            hist[g].append((boards[g][:], players[g], pi))
            boards[g][a] = players[g]
            players[g] = -players[g]
            t = term(boards[g])
            if t is None:
                still.append(g)
            else:
                out += [(bb, p, pi_, t * p) for bb, p, pi_ in hist[g]]
        active = still
    return out


torch.manual_seed(0)
print(len(batched_self_play(Net(), 8, 20, np.random.default_rng(0))))
