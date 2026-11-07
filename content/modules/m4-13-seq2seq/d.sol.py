import random
import torch
import torch.nn as nn
import torch.nn.functional as F

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
IN = sorted(set("".join(MONTHS) + "0123456789, "))
OUT = list("0123456789-")
PAD, SOS, EOS = 0, 1, 2


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


class Seq2Seq(nn.Module):
    def __init__(self, h=64, attn=True):
        super().__init__()
        self.attn = attn
        V = len(OUT) + 3
        self.emb = nn.Embedding(len(IN) + 1, 32, padding_idx=PAD)
        self.enc = nn.GRU(32, h, batch_first=True, bidirectional=True)
        self.demb = nn.Embedding(V, 32)
        self.init = nn.Linear(2 * h, h)
        self.cell = nn.GRUCell(32 + (2 * h if attn else 0), h)
        self.W1 = nn.Linear(2 * h, h, bias=False)
        self.W2 = nn.Linear(h, h)
        self.v = nn.Linear(h, 1, bias=False)
        self.out = nn.Linear(h + (2 * h if attn else 0), V)

    def _encode(self, x):
        H, hn = self.enc(self.emb(x))
        return H, torch.tanh(self.init(torch.cat([hn[0], hn[1]], -1))), x != PAD

    def _step(self, tok, h, H, mask, keys):
        e = self.demb(tok)
        if not self.attn:
            h = self.cell(e, h)
            return self.out(h), h, None
        sc = self.v(torch.tanh(keys + self.W2(h)[:, None])).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = sc.softmax(-1)
        ctx = (a[..., None] * H).sum(1)
        h = self.cell(torch.cat([e, ctx], -1), h)
        return self.out(torch.cat([h, ctx], -1)), h, a

    def forward(self, x, y):
        H, h, mask = self._encode(x)
        keys = self.W1(H)
        tok = torch.full((len(x),), SOS)
        outs = []
        for t in range(y.shape[1]):
            o, h, _ = self._step(tok, h, H, mask, keys)
            outs.append(o)
            tok = y[:, t]
        return torch.stack(outs, 1)

    @torch.no_grad()
    def _decode(self, x, T):
        H, h, mask = self._encode(x)
        keys = self.W1(H)
        tok = torch.full((len(x),), SOS)
        toks, attn = [], []
        for _ in range(T):
            o, h, a = self._step(tok, h, H, mask, keys)
            tok = o.argmax(-1)
            toks.append(tok)
            attn.append(a)
        return torch.stack(toks, 1), (torch.stack(attn, 1) if self.attn else None)

    @torch.no_grad()
    def greedy(self, x, T=11):
        return self._decode(x, T)[0]

    @torch.no_grad()
    def attention_map(self, x, T=11):
        return self._decode(x, T)[1]
