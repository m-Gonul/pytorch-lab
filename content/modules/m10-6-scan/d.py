import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def scan(x, BLOCK=1024):
    ...


def linear_recurrence(a, b):
    ...


print(scan(torch.ones(3000))[-3:])
