import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
IN = sorted(set("".join(MONTHS) + "0123456789, "))
OUT = list("0123456789-")
PAD, SOS, EOS = 0, 1, 2
V_IN, V_OUT = len(IN) + 1, len(OUT) + 3


def dates(n, seed):
    r = random.Random(seed)
    X, Y = [], []
    for _ in range(n):
        y, m, d = r.randint(1000, 2999), r.randint(1, 12), r.randint(1, 28)
        X.append([f"{MONTHS[m - 1]} {d}, {y}", f"{d} {MONTHS[m - 1]} {y}", f"{MONTHS[m - 1][:3]} {d:02d} {y}"][r.randint(0, 2)])
        Y.append(f"{y}-{m:02d}-{d:02d}")
    return X, Y


def encode(X, Y):
    iv = {c: i + 1 for i, c in enumerate(IN)}
    ov = {c: i + 3 for i, c in enumerate(OUT)}
    L = max(map(len, X))
    xs = torch.tensor([[iv[c] for c in x] + [PAD] * (L - len(x)) for x in X])
    ys = torch.tensor([[ov[c] for c in y] + [EOS] for y in Y])
    return xs, ys


class EncoderLayer(nn.Module):
    def __init__(self, d, h, ff, dropout=0.0):
        super().__init__()
        ...

    def forward(self, x, pad):
        ...


class DecoderLayer(nn.Module):
    def __init__(self, d, h, ff, dropout=0.0):
        super().__init__()
        ...

    def forward(self, y, mem, causal, mem_pad):
        ...


class Transformer(nn.Module):
    def __init__(self, d=64, h=4, ff=128, N=2, dropout=0.1, emb_std=None):
        super().__init__()
        ...

    def encode(self, x):
        ...

    def decode(self, yin, mem, pad):
        ...

    def forward(self, x, yin):
        ...

    @torch.no_grad()
    def greedy(self, x, T=11):
        ...
