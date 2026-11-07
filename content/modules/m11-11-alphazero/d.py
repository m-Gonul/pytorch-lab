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
        ...

    def forward(self, x):
        ...


def encode(board, player):
    ...


def mcts(net, board, player, n_sims, rng, c=1.5, noise=True):
    ...


def self_play(net, n_sims, rng):
    ...


def train(net, opt, data, epochs=3, bs=64):
    ...


def alphazero(iters=15, games=20, n_sims=30, seed=0):
    ...


net = alphazero(2, 5, 10)
print(mcts(net, [0] * 9, 1, 30, np.random.default_rng(0), noise=False))
