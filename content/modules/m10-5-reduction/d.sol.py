import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def _bsum(X, OUT, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + tl.program_id(0), tl.sum(tl.load(X + offs, mask=offs < n, other=0.0), axis=0))


def block_sums(x, BLOCK=1024):
    n = x.numel()
    out = torch.empty(triton.cdiv(n, BLOCK), dtype=torch.float32, device=x.device)
    _bsum[(out.numel(),)](x, out, n, BLOCK=BLOCK)
    return out


def sum_two_pass(x):
    while x.numel() > 1:
        x = block_sums(x)
    return x[0]


@triton.jit
def _asum(X, OUT, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.atomic_add(OUT, tl.sum(tl.load(X + offs, mask=offs < n, other=0.0), axis=0))


def sum_atomic(x):
    out = torch.zeros(1, dtype=torch.float32, device=x.device)
    _asum[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    return out[0]


@triton.jit
def _amax(X, V, I, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < n, other=-float("inf"))
    tl.store(V + pid, tl.max(x, axis=0))
    tl.store(I + pid, pid * BLOCK + tl.argmax(x, axis=0))


def argmax(x):
    nb = triton.cdiv(x.numel(), 1024)
    V = torch.empty(nb, dtype=torch.float32, device=x.device)
    I = torch.empty(nb, dtype=torch.int64, device=x.device)
    _amax[(nb,)](x, V, I, x.numel(), BLOCK=1024)
    return int(I[int(torch.argmax(V))])


@triton.jit
def _hist(X, H, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(X + offs, mask=m, other=0)
    tl.atomic_add(H + v, 1, mask=m)


def histogram(x, nbins):
    h = torch.zeros(nbins, dtype=torch.int32, device=x.device)
    _hist[(triton.cdiv(x.numel(), 1024),)](x, h, x.numel(), BLOCK=1024)
    return h


def butterfly(x):
    lane = torch.arange(32, device=x.device)
    for off in (16, 8, 4, 2, 1):
        x = x + x[:, lane ^ off]
    return x


print(butterfly(torch.arange(64.0).reshape(2, 32))[:, :3])
