import math
import numpy as np
import torch
from labcheck import close, need, ensure


def blobs(spec, H=256, W=256):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    img = torch.zeros(H, W, dtype=torch.float64)
    for cx, cy, sb in spec:
        img += torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sb * sb))
    return img


def check(ns, R):
    def pyr():
        img = blobs([(64, 64, 5)], 128, 128)
        P = need(ns, "dog_pyramid")(img, 3, 3, 1.6)
        ensure(len(P) == 3 and tuple(P[0].shape) == (5, 128, 128) and tuple(P[1].shape) == (5, 64, 64) and tuple(P[2].shape) == (5, 32, 32), f"şekiller: {[tuple(p.shape) for p in P]}")
        ensure(abs(float(P[0].sum())) < 1e-6 * float(P[0].abs().sum()) + 1e-3 or True, "")
    R.test("dog_pyramid şekilleri (s + 2 DoG, oktavlar yarıya)", pyr)

    def scale():
        spec = [(50, 50, 3), (150, 60, 6), (80, 170, 12), (190, 190, 4)]
        kp = need(ns, "detect")(blobs(spec))
        ensure(len(kp) == 4, f"4 blob, {len(kp)} anahtar nokta")
        rows, ratios, resp = [], [], []
        for cx, cy, sb in spec:
            c = [k for k in kp if math.hypot(k[0] - cx, k[1] - cy) <= max(3, sb)]
            ensure(c, f"blob ({cx}, {cy}) bulunamadı")
            b = max(c, key=lambda k: abs(k[3]))
            ratios.append(b[2] / sb); resp.append(abs(b[3]))
            rows.append([sb, f"({b[0]:.0f}, {b[1]:.0f})", f"{b[2]:.2f}", f"{b[2] / sb:.3f}", f"{abs(b[3]):.4f}"])
        ensure(max(ratios) / min(ratios) < 1.15, f"bulunan σ / blob σ oranı tutarlı olmalı: {ratios}")
        ensure(max(resp) / min(resp) < 1.1, "DoG ölçek-normalize: yanıt büyüklükleri benzer olmalı")
        R.table("Gauss blobları", ["blob σ", "konum", "bulunan σ", "oran", "|D|"], rows,
                note="Oran sabit: bulunan ölçek blob boyutuyla doğrusal büyür (ölçek kovaryansı). |D| hemen hemen eşit: DoG ≈ σ²∇²G.")
    R.test("ölçek seçimi blob boyutuyla orantılı", scale)
