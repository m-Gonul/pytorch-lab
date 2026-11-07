import time
import torch
import torch.nn as nn


def time_blocks(C, H, reps=5):
    x = torch.randn(8, C, H, H)
    blocks = {
        "3×3 standart": (nn.Conv2d(C, C, 3, padding=1, bias=False), 9 * C * C),
        "3×3 ayrılabilir": (nn.Sequential(nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False), nn.Conv2d(C, C, 1, bias=False)), 9 * C + C * C),
        "3×3 grup (g = 8)": (nn.Conv2d(C, C, 3, padding=1, groups=8, bias=False), 9 * C * C // 8),
        "1×1": (nn.Conv2d(C, C, 1, bias=False), C * C),
    }
    out = {}
    with torch.no_grad():
        for name, (m, per_px) in blocks.items():
            m(x)
            best = 1e9
            for _ in range(reps):
                t0 = time.perf_counter()
                m(x)
                best = min(best, time.perf_counter() - t0)
            macs = 8 * H * H * per_px
            out[name] = (macs, best, macs / best / 1e9)
    return out
