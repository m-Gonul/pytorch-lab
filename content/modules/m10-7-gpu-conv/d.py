import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def conv2d(img, k, BH=16, BW=16):
    ...


def matmul(a, b, BM=32, BN=32, BK=32, GROUP_M=4):
    ...


print(conv2d(torch.full((8, 8), 10, dtype=torch.uint8), torch.ones(3, 3) / 9))
