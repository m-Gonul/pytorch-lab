import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def copy_kernel(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(Y + offs, tl.load(X + offs, mask=m), mask=m)


def copy(x):
    y = torch.empty_like(x)
    n = x.numel()
    copy_kernel[(triton.cdiv(n, 256),)](x, y, n, BLOCK=256)
    return y


@triton.jit
def tnaive_kernel(X, Y, M, N, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < N
    v = tl.load(X + r * N + c, mask=m)
    tl.store(Y + c * M + r, v, mask=m)


def transpose_naive(x):
    M, N = x.shape
    x = x.contiguous()
    y = torch.empty(N, M, dtype=x.dtype, device=x.device)
    tnaive_kernel[(M, triton.cdiv(N, 64))](x, y, M, N, BLOCK=64)
    return y


@triton.jit
def ttiled_kernel(X, Y, M, N, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    blk = tl.load(X + rm[:, None] * N + rn[None, :], mask=m)
    tl.store(Y + rn[:, None] * M + rm[None, :], tl.trans(blk), mask=tl.trans(m))


def transpose_tiled(x, BM=16, BN=16):
    M, N = x.shape
    x = x.contiguous()
    y = torch.empty(N, M, dtype=x.dtype, device=x.device)
    ttiled_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](x, y, M, N, BM=BM, BN=BN)
    return y


print(copy(torch.arange(10.0)))
