import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, need, ensure


def entropy(img_u8, bins=16):
    p = np.bincount(((img_u8.astype(np.int64) * bins) // 256).ravel(), minlength=bins) / img_u8.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def check(ns, R):
    x = torch.tensor(data.moon(), dtype=torch.float32) / 255

    def sh():
        v = torch.tensor([0.15, 0.15, 0.95], requires_grad=True)
        h = need(ns, "soft_histogram")(v, bins=10)
        close(h.sum(), 1.0)
        ensure(h.argmax().item() == 1 and h[1] > h[9] > h[5], "tepe 2. kutuda, ikinci tepe sonuncuda")
        (h[1]).backward()
        ensure(v.grad is not None and v.grad.abs().sum() > 0, "gradyan akmalı")
    R.test("soft_histogram: normalize ve türevlenebilir", sh)

    def tc():
        c = need(ns, "ToneCurve", "sınıf")()
        y = c(torch.linspace(0, 1, 101))
        close(y[0], 0.0, atol=1e-6); close(y[-1], 1.0, atol=1e-6)
        close(y, torch.linspace(0, 1, 101), atol=1e-5, what="başlangıçta özdeşlik")
    R.test("ToneCurve: başlangıçta özdeşlik, uçlar sabit", tc)

    def learn():
        torch.manual_seed(0)
        curve = need(ns, "learn_equalization")(x)
        with torch.no_grad():
            grid = curve(torch.linspace(0, 1, 256))
            out = (curve(x) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
        ensure(bool((grid[1:] >= grid[:-1] - 1e-7).all()), "eğri monoton olmalı")
        e0, e1, e2 = entropy(data.moon()), entropy(out), entropy(cv2.equalizeHist(data.moon()))
        ensure(e1 > e0 + 1.0 and e1 > 0.95 * e2, f"16 kutulu entropi: orijinal {e0:.2f}, öğrenilen {e1:.2f}, cv2 HE {e2:.2f} bit")
        return f"16 kutulu histogram entropisi (bit, en fazla 4): orijinal {e0:.2f} → öğrenilen eğri {e1:.2f} (cv2 HE {e2:.2f})"
    R.test("öğrenilen ton eğrisi histogramı düzleştiriyor", learn)
