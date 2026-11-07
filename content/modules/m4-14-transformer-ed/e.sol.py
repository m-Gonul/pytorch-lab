import time
import torch


@torch.no_grad()
def greedy_naive(model, x, T):
    y = torch.full((len(x), 1), 1)
    for _ in range(T):
        y = torch.cat([y, model(x, y)[:, -1].argmax(-1, keepdim=True)], 1)
    return y[:, 1:]


@torch.no_grad()
def greedy_cached(model, x, T):
    mem, pad = model.encode(x)
    y = torch.full((len(x), 1), 1)
    for _ in range(T):
        y = torch.cat([y, model.decode(y, mem, pad)[:, -1].argmax(-1, keepdim=True)], 1)
    return y[:, 1:]
