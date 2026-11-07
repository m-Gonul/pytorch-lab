import math
import torch
import torch.nn.functional as F
from torch import nn


class BigramLM(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.table = nn.Embedding(V, V)

    def forward(self, idx):
        return self.table(idx)


def train_bigram(ids, V, steps=500, lr=0.5):
    model = BigramLM(V)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = ids[:-1], ids[1:]
    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, loss.item()


def count_nll(ids, V):
    N = torch.zeros(V, V).index_put_((ids[:-1], ids[1:]), torch.ones(len(ids) - 1), accumulate=True)
    P = N / N.sum(1, keepdim=True).clamp_min(1)
    return -P[ids[:-1], ids[1:]].log().mean().item()


text = "kedi kediye bakti. kopek kediye havladi. kedi kacti. " * 20
chars = sorted(set(text))
ids = torch.tensor([chars.index(c) for c in text])
model, loss = train_bigram(ids, len(chars))
print(loss, count_nll(ids, len(chars)))
