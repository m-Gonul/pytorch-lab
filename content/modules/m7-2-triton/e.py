import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def matmul_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ACT: tl.constexpr):
    ...


def matmul(a, b, activation=None):
    ...


a, b = torch.randn(70, 50, device=DEVICE), torch.randn(50, 40, device=DEVICE)
print((matmul(a, b, "relu") - torch.relu(a @ b)).abs().max())
