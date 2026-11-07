import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")   # GPU yoksa CPU'da yorumlayıcı
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    ...


def add(x, y):
    ...


@triton.jit
def softmax_kernel(X, Y, stride, n_cols, BLOCK: tl.constexpr):
    ...


def softmax(x):
    ...


x = torch.randn(1000, device=DEVICE)
print((add(x, x) - 2 * x).abs().max())
