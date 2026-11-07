import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def copy_kernel(X, Y, n, BLOCK: tl.constexpr):
    pass  # TODO


def copy(x):
    ...


def transpose_naive(x):
    ...


def transpose_tiled(x, BM=16, BN=16):
    ...


print(copy(torch.arange(10.0)))
