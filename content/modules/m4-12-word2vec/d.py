import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def embedding_as_linear(emb):
    ...


class SGNS(nn.Module):
    def __init__(self, V, d, sparse=False):
        super().__init__()
        ...

    def forward(self, c, o, neg):
        ...


class CBOW(nn.Module):
    def __init__(self, V, d, pad):
        super().__init__()
        ...

    def forward(self, ctx, o, neg):
        ...


def train(model, batches, epochs, lr, sparse=False):
    ...
