import numpy as np
import cv2
from labcheck import close, need, ensure


def check(ns, R):
    rng = np.random.default_rng(0)
    Ht = np.array([[1.02, 0.03, 40], [-0.02, 0.98, 25], [2e-5, -1e-5, 1]])

    def t():
        er, en, cr, cn = [], [], [], []
        for k in range(200):
            src = rng.uniform(0, 1000, (8, 2))
            dst = cv2.perspectiveTransform(src[None], Ht)[0] + rng.normal(0, 1.0, (8, 2))
            test = rng.uniform(0, 1000, (100, 2))
            want = cv2.perspectiveTransform(test[None], Ht)[0]
            for H, acc in ((need(ns, "dlt_raw")(src, dst), er), (need(ns, "dlt_norm")(src, dst), en)):
                acc.append(np.linalg.norm(cv2.perspectiveTransform(test[None], H)[0] - want, axis=1).mean())
            cr.append(need(ns, "cond_A")(src, dst, False)); cn.append(ns["cond_A"](src, dst, True))
        ensure(np.median(cn) * 1e4 < np.median(cr), "normalizasyon koşul sayısını büyük ölçüde düşürmeli")
        ensure(np.mean(en) < np.mean(er), "normalize DLT görülmemiş noktalarda daha doğru olmalı")
        R.table("200 deneme, 8 eş, σ = 1 piksel, koordinatlar 0–1000", ["DLT", "A'nın koşul sayısı (medyan)", "görülmemiş noktalarda hata (px)"],
                [["normalizasyonsuz", f"{np.median(cr):.1e}", f"{np.mean(er):.2f}"], ["Hartley normalizasyonlu", f"{np.median(cn):.1e}", f"{np.mean(en):.2f}"]])
    R.test("normalizasyonun etkisi", t)
