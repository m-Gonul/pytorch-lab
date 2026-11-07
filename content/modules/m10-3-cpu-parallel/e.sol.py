import os
import time
import torch
import torch.nn.functional as F


def _best(f, r=3):
    b = 1e9
    for _ in range(r):
        t0 = time.perf_counter()
        f()
        b = min(b, time.perf_counter() - t0)
    return b


def compare(ext, threads):
    g = torch.Generator().manual_seed(1)
    x = torch.randint(-1000, 1000, (4_000_000,), dtype=torch.int32, generator=g)
    img = torch.randint(0, 256, (1024, 1024), dtype=torch.uint8, generator=g)
    k = torch.ones(11, 11) / 121
    A = torch.randn(512, 512, generator=g)
    B = torch.randn(512, 512, generator=g)
    out = {}
    out["integral dizi"] = (_best(lambda: ext.integral_array(x, 1)), _best(lambda: ext.integral_array(x, threads)), _best(lambda: torch.cumsum(x, 0)))
    out["integral görüntü"] = (_best(lambda: ext.integral_image(img, 1)), _best(lambda: ext.integral_image(img, threads)), _best(lambda: img.long().cumsum(0).cumsum(1)))
    out["konvolüsyon 11×11"] = (_best(lambda: ext.conv2d(img, k, 1), 2), _best(lambda: ext.conv2d(img, k, threads)), _best(lambda: F.conv2d(img.float()[None, None], k[None, None], padding=5)))
    out["matmul 512"] = (_best(lambda: ext.matmul(A, B, 1, "tiled"), 2), _best(lambda: ext.matmul(A, B, threads, "tiled")), _best(lambda: A @ B))
    return out


def size_sweep(ext, sizes, threads):
    rows = []
    g = torch.Generator().manual_seed(2)
    for s in sizes:
        img = torch.randint(0, 256, (s, s), dtype=torch.uint8, generator=g)
        r = 20 if s <= 256 else 3
        t1 = _best(lambda: ext.integral_image(img, 1), r)
        tp = _best(lambda: ext.integral_image(img, threads), r)
        rows.append((s, t1, tp, t1 / tp))
    return rows
