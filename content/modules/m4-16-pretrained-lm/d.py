import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, CLS, SEP, MASK = 0, 1, 2, 3
V = 64


def corpus(n, seed, T=16, K=5):
    g = torch.Generator().manual_seed(seed)
    topic = torch.randint(0, K, (n,), generator=g)
    content = 24 + topic[:, None] * 8 + torch.randint(0, 8, (n, T // 2), generator=g)
    noise = torch.rand(n, T // 2, generator=g) < 0.5
    content = torch.where(noise, 24 + torch.randint(0, K * 8, (n, T // 2), generator=g), content)
    func = 4 + (content * 7 + torch.arange(T // 2)) % 20
    seq = torch.stack([func, content], 2).flatten(1)
    return torch.cat([torch.full((n, 1), CLS), seq], 1), topic


def mlm_mask(x, g, V, p=0.15):
    ...


class TinyBERT(nn.Module):
    def __init__(self, V, d=64, L=2):
        super().__init__()
        ...

    def forward(self, x):
        ...


def pretrain(model, U, epochs, lr, seed):
    ...


def finetune(model, X, y, steps, lr, freeze=False):
    ...
