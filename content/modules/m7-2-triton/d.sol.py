import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")   # GPU yoksa CPU'da yorumlayıcı
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    return out


@triton.jit
def softmax_kernel(X, Y, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(Y + row * stride + cols, e / tl.sum(e, axis=0), mask=mask)


def softmax(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    softmax_kernel[(x.shape[0],)](x, y, x.stride(0), x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]))
    return y


x = torch.randn(1000, device=DEVICE)
print((add(x, x) - 2 * x).abs().max())
