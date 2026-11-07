import os
import socket
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def make_model():
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))


def make_data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(64, 8, generator=g), torch.randn(64, 1, generator=g)


# ---- senin kodun ----

def allreduce_mean(t):
    ...


def worker(rank, world, port, out, steps, manual):
    ...


def run(world, steps=5, manual=False):
    ...


P = run(2)
print(all(torch.equal(a, b) for a, b in zip(P[0], P[1])))
