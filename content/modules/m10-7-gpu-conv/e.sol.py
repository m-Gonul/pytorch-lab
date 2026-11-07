import time
import torch
import torch.nn.functional as F


def conv_direct(img, k):
    return F.conv2d(img[None, None], k[None, None], padding=k.shape[0] // 2)[0, 0]


def conv_unfold(img, k):
    H, W = img.shape
    K = k.shape[0]
    cols = F.unfold(img[None, None], K, padding=K // 2)[0]
    return (k.reshape(1, -1) @ cols).reshape(H, W)


def conv_fft(img, k):
    H, W = img.shape
    K = k.shape[0]
    s = (H + K - 1, W + K - 1)
    Fr = torch.fft.rfft2(img, s) * torch.fft.rfft2(torch.flip(k, (0, 1)), s)
    r = K // 2
    return torch.fft.irfft2(Fr, s)[r:r + H, r:r + W]


def _best(f, r=3):
    b = 1e9
    for _ in range(r):
        t0 = time.perf_counter()
        f()
        b = min(b, time.perf_counter() - t0)
    return b


def crossover(H, Ks):
    g = torch.Generator().manual_seed(0)
    img = torch.rand(H, H, generator=g) * 255
    out = []
    for K in Ks:
        k = torch.rand(K, K, generator=g)
        ref = conv_direct(img, k)
        d1 = float((conv_unfold(img, k) - ref).abs().max())
        d2 = float((conv_fft(img, k) - ref).abs().max())
        out.append((K, _best(lambda: conv_direct(img, k)), _best(lambda: conv_unfold(img, k)), _best(lambda: conv_fft(img, k)), max(d1, d2) / float(ref.abs().max())))
    return out
