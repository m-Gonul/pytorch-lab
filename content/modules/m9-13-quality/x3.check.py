import numpy as np
import cv2
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    base = cv2.GaussianBlur(data.camera().astype(np.float64), (0, 0), 2)

    def nz():
        rows = []
        for s in (5, 10, 20):
            img = base + np.random.default_rng(s).normal(0, s, base.shape)
            e = need(ns, "estimate_noise")(img)
            ensure(abs(e - s) / s < 0.15, f"σ = {s}: kestirim {e:.2f}")
            rows.append([s, f"{e:.2f}"])
        R.table("Immerkær kestirimi (yumuşak görüntü + Gauss gürültüsü)", ["gerçek σ", "kestirim"], rows, note="Dokulu görüntülerde kenarlar kestirimi şişirir; bu yüzden yöntemler genelde düz bölgeleri seçer.")
    R.test("estimate_noise", nz)

    def bl():
        img = data.camera()
        vals = [need(ns, "blur_score")(cv2.GaussianBlur(img, (0, 0), s) if s else img) for s in (0, 1, 2, 4)]
        ensure(all(a > b for a, b in zip(vals, vals[1:])), f"bulanıklaştıkça azalmalı: {vals}")
        return " → ".join(f"{v:.0f}" for v in vals) + " (σ = 0, 1, 2, 4)"
    R.test("blur_score", bl)
