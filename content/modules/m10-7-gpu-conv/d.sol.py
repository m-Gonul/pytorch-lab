import os
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def _conv(X, Kp, Y, H, W, K: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr):
    ry = tl.program_id(0) * BH + tl.arange(0, BH)
    rx = tl.program_id(1) * BW + tl.arange(0, BW)
    acc = tl.zeros((BH, BW), dtype=tl.float32)
    for u in range(K):
        for v in range(K):
            yy = ry[:, None] + (u - K // 2)
            xx = rx[None, :] + (v - K // 2)
            m = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            x = tl.load(X + yy * W + xx, mask=m, other=0).to(tl.float32)
            acc += tl.load(Kp + u * K + v) * x
    q = tl.minimum(tl.maximum((acc + 0.5).to(tl.int32), 0), 255)
    om = (ry[:, None] < H) & (rx[None, :] < W)
    tl.store(Y + ry[:, None] * W + rx[None, :], q.to(tl.uint8), mask=om)


def conv2d(img, k, BH=16, BW=16):
    img = img.contiguous()
    k = k.float().contiguous().to(img.device)
    H, W = img.shape
    y = torch.empty_like(img)
    _conv[(triton.cdiv(H, BH), triton.cdiv(W, BW))](img, k, y, H, W, K=k.shape[0], BH=BH, BW=BW)
    return y


@triton.jit
def _mm(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BM)
    n_n = tl.cdiv(N, BN)
    group = GROUP_M * n_n
    first_m = (pid // group) * GROUP_M
    gsize = tl.minimum(n_m - first_m, GROUP_M)
    pm = first_m + (pid % group) % gsize
    pn = (pid % group) // gsize
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(A + rm[:, None] * K + (k0 + rk)[None, :], mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        b = tl.load(B + (k0 + rk)[:, None] * N + rn[None, :], mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a, b, BM=32, BN=32, BK=32, GROUP_M=4):
    a, b = a.float().contiguous(), b.float().contiguous()
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, dtype=torch.float32, device=a.device)
    _mm[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK, GROUP_M=GROUP_M)
    return c


print(conv2d(torch.full((8, 8), 10, dtype=torch.uint8), torch.ones(3, 3) / 9))
