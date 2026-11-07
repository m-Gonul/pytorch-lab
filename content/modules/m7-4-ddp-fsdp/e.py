import os
import socket
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))


def make_data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(64, 8, generator=g), torch.randn(64, 1, generator=g)


# ---- senin kodun ----

def flatten(ts):
    ...


def unflatten(flat, like):
    ...


def zero1_worker(rank, world, port, out, steps, lr=1e-2):
    ...


def run_zero1(world, steps=10):
    ...


P, sizes = run_zero1(2)
print(sizes)
