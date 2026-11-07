import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def block_sums(x, BLOCK=1024):
    ...


def sum_two_pass(x):
    ...


def sum_atomic(x):
    ...


def argmax(x):
    ...


def histogram(x, nbins):
    ...


def butterfly(x):
    ...


print(butterfly(torch.arange(64.0).reshape(2, 32))[:, :3])
