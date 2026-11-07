import numpy as np
import cv2
import torch
from skimage import data
from labcheck import need, ensure


def video(T=80, seed=0, sigma=6):
    rng = np.random.default_rng(seed)
    bg = data.camera()[::3, ::3][:120, :160].astype(np.float64)
    obj = data.coins()[40:80, 40:80].astype(np.float64)
    frames, gts = [], []
    for t in range(T):
        f = bg.copy(); gt = np.zeros(bg.shape, bool)
        f = f + 4 * np.sin(np.arange(160) / 6 + t / 3)[None, :] * (np.arange(120) < 20)[:, None]
        x = 5 + 2 * t
        if 30 <= t and x + 30 < 160:
            f[40:70, x:x + 30] = obj[:30, :30]; gt[40:70, x:x + 30] = True
        frames.append(np.clip(f + rng.normal(0, sigma, f.shape), 0, 255).astype(np.uint8)); gts.append(gt)
    return np.stack(frames), np.stack(gts)


def f1(pred, gt):
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def check(ns, R):
    frames, gts = video()

    def unit():
        g = need(ns, "GMM", "sınıf")((4, 5))
        f = torch.full((4, 5), 100.0, dtype=torch.float64)
        ensure(not g.apply(f).any(), "ilk kare arka plan")
        for _ in range(5):
            ensure(not g.apply(f + 3).any(), "küçük değişim arka plan")
        ensure(bool(g.apply(f + 120).all()), "büyük değişim ön plan")
        ensure(torch.allclose(g.w.sum(0), torch.ones(4, 5, dtype=torch.float64)), "ağırlıklar normalize")
    R.test("GMM birim davranışı", unit)

    def vid():
        g = ns["GMM"](frames.shape[1:], alpha=0.02)
        pred = np.stack([g.apply(torch.tensor(f, dtype=torch.float64)).numpy() for f in frames])
        mog = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
        pm = np.stack([mog.apply(f) > 0 for f in frames])
        a, b = f1(pred[30:], gts[30:]), f1(pm[30:], gts[30:])
        ensure(a > 0.8, f"F1 {a:.3f}")
        R.table("Sentetik video (üst şeritte dalgalanan arka plan, nesne 30. karede giriyor)", ["yöntem", "F1 (kare 30+)"], [["senin GMM (K = 3)", f"{a:.3f}"], ["OpenCV MOG2", f"{b:.3f}"]])
    R.test("videoda GMM", vid)
