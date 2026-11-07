import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def _block_scan(X, Y, S, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < n, other=0.0)
    y = tl.cumsum(x, axis=0)
    tl.store(Y + offs, y, mask=offs < n)
    tl.store(S + pid, tl.sum(x, axis=0))


@triton.jit
def _add_offsets(Y, P, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0) + 1
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(Y + offs, tl.load(Y + offs, mask=m) + tl.load(P + pid - 1), mask=m)


def scan(x, BLOCK=1024):
    n = x.numel()
    nb = triton.cdiv(n, BLOCK)
    y = torch.empty_like(x)
    sums = torch.empty(nb, dtype=x.dtype, device=x.device)
    _block_scan[(nb,)](x, y, sums, n, BLOCK=BLOCK)
    if nb > 1:
        pref = scan(sums, BLOCK)
        _add_offsets[(nb - 1,)](y, pref, n, BLOCK=BLOCK)
    return y


@triton.jit
def _affine(a1, b1, a2, b2):
    return a2 * a1, a2 * b1 + b2


@triton.jit
def _linrec(A, B, X, L, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < L
    a = tl.load(A + row * L + offs, mask=m, other=1.0)
    b = tl.load(B + row * L + offs, mask=m, other=0.0)
    _, x = tl.associative_scan((a, b), 0, _affine)
    tl.store(X + row * L + offs, x, mask=m)


def linear_recurrence(a, b):
    a, b = a.contiguous(), b.contiguous()
    B, L = a.shape
    x = torch.empty_like(b)
    _linrec[(B,)](a, b, x, L, BLOCK=triton.next_power_of_2(L))
    return x


print(scan(torch.ones(3000))[-3:])
