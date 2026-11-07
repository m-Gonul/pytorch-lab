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


def _pe(n, d):
    p = torch.arange(n)[:, None].float()
    w = 10000.0 ** (-torch.arange(0, d, 2).float() / d)
    P = torch.zeros(n, d)
    P[:, 0::2] = torch.sin(p * w)
    P[:, 1::2] = torch.cos(p * w)
    return P


class EncoderLayer(nn.Module):
    def __init__(self, d, h, ff, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.linear1, self.linear2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, key_padding_mask=pad, need_weights=False)[0]))
        return self.norm2(x + self.drop(self.linear2(self.drop(F.relu(self.linear1(x))))))


class DecoderLayer(nn.Module):
    def __init__(self, d, h, ff, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.linear1, self.linear2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, y, mem, causal, mem_pad):
        y = self.norm1(y + self.drop(self.self_attn(y, y, y, attn_mask=causal, need_weights=False)[0]))
        y = self.norm2(y + self.drop(self.multihead_attn(y, mem, mem, key_padding_mask=mem_pad, need_weights=False)[0]))
        return self.norm3(y + self.drop(self.linear2(self.drop(F.relu(self.linear1(y))))))


class Transformer(nn.Module):
    def __init__(self, d=64, h=4, ff=128, N=2, dropout=0.1, emb_std=None):
        super().__init__()
        self.d = d
        self.src, self.tgt = nn.Embedding(V_IN, d), nn.Embedding(V_OUT, d)
        std = d ** -0.5 if emb_std is None else emb_std
        nn.init.normal_(self.src.weight, 0, std)
        nn.init.normal_(self.tgt.weight, 0, std)
        self.register_buffer("P", _pe(128, d))
        self.enc = nn.ModuleList([EncoderLayer(d, h, ff, dropout) for _ in range(N)])
        self.dec = nn.ModuleList([DecoderLayer(d, h, ff, dropout) for _ in range(N)])
        self.out = nn.Linear(d, V_OUT)
        self.drop = nn.Dropout(dropout)

    def encode(self, x):
        pad = x == PAD
        m = self.drop(self.src(x) * math.sqrt(self.d) + self.P[:x.shape[1]])
        for layer in self.enc:
            m = layer(m, pad)
        return m, pad

    def decode(self, yin, mem, pad):
        T = yin.shape[1]
        causal = torch.triu(torch.full((T, T), float("-inf")), 1)
        y = self.drop(self.tgt(yin) * math.sqrt(self.d) + self.P[:T])
        for layer in self.dec:
            y = layer(y, mem, causal, pad)
        return self.out(y)

    def forward(self, x, yin):
        mem, pad = self.encode(x)
        return self.decode(yin, mem, pad)

    @torch.no_grad()
    def greedy(self, x, T=11):
        mem, pad = self.encode(x)
        y = torch.full((len(x), 1), SOS)
        for _ in range(T):
            y = torch.cat([y, self.decode(y, mem, pad)[:, -1].argmax(-1, keepdim=True)], 1)
        return y[:, 1:]
