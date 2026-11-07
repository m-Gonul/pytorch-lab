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
    ...


def batched_self_play(net, n_games, n_sims, rng, c=1.5):
    ...


torch.manual_seed(0)
print(len(batched_self_play(Net(), 8, 20, np.random.default_rng(0))))
