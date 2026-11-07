import math
import torch
import torch.nn.functional as F
from torch import nn


class BigramLM(nn.Module):
    def __init__(self, V):
        super().__init__()
        ...

    def forward(self, idx):
        ...


def train_bigram(ids, V, steps=500, lr=0.5):
    ...


def count_nll(ids, V):
    ...


text = "kedi kediye bakti. kopek kediye havladi. kedi kacti. " * 20
chars = sorted(set(text))
ids = torch.tensor([chars.index(c) for c in text])
model, loss = train_bigram(ids, len(chars))
print(loss, count_nll(ids, len(chars)))
