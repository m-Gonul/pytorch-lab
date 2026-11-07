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
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:, None] * sam + (k + rk)[None, :] * sak,
                    mask=(rm[:, None] < M) & ((k + rk)[None, :] < K), other=0.0)
        b = tl.load(B + (k + rk)[:, None] * sbk + rn[None, :] * sbn,
                    mask=((k + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    if ACT == "relu":
        acc = tl.maximum(acc, 0.0)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a, b, activation=None):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    BM, BN, BK = 32, 32, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BM=BM, BN=BN, BK=BK, ACT=activation or "none")
    return c


a, b = torch.randn(70, 50, device=DEVICE), torch.randn(50, 40, device=DEVICE)
print((matmul(a, b, "relu") - torch.relu(a @ b)).abs().max())
