import numpy as np
import cv2
import torch
from labcheck import max_loops, need, ensure, bench, fmt_time


def scene(seed, n=6, S=48, up=8):
    rng = np.random.default_rng(seed)
    H = W = S * n
    hi = np.full((H * up, W * up), 40, np.uint8)
    pts = []
    for i in range(n):
        for j in range(n):
            c = np.array([j * S + S / 2, i * S + S / 2]); k = rng.integers(3, 5)
            ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(0, 2 * np.pi)
            P = c + rng.uniform(13, 17) * np.stack([np.cos(ang), np.sin(ang)], 1)
            X = (P + 0.5) * up - 0.5
            cv2.fillPoly(hi, [np.round(X * 16).astype(np.int32)], int(rng.integers(150, 230)), lineType=cv2.LINE_8, shift=4)
            pts.append(P)
    return cv2.resize(hi, (W, H), interpolation=cv2.INTER_AREA), np.concatenate(pts)


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız iterasyon döngüsü", lambda: max_loops(src, 1, func="refine_subpix"))

    def acc():
        rows = []
        for seed in (0, 1):
            img, P = scene(seed)
            init = np.round(P + np.random.default_rng(seed + 10).uniform(-1, 1, P.shape))
            got = np.asarray(need(ns, "refine_subpix")(img, init), np.float64)
            ensure(got.shape == P.shape, "(N, 2)")
            e0, e1 = np.linalg.norm(init - P, axis=1).mean(), np.linalg.norm(got - P, axis=1).mean()
            c = init.astype(np.float32).reshape(-1, 1, 2).copy()
            cv2.cornerSubPix(img, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
            ec = np.linalg.norm(c[:, 0] - P, axis=1).mean()
            ensure(e1 < 0.25 and e1 < e0 / 3, f"ortalama hata {e1:.3f} piksel (başlangıç {e0:.3f})")
            ensure(e1 < ec + 0.05, f"senin {e1:.3f}, cv2.cornerSubPix {ec:.3f}")
            rows.append([seed, len(P), f"{e0:.3f}", f"{e1:.3f}", f"{ec:.3f}"])
        R.table("Alt piksel köşe hatası (piksel)", ["sahne", "köşe", "tamsayı başlangıç", "senin", "cv2.cornerSubPix"], rows,
                note="Kalan ~0.15 piksel, 8× üst örneklemeyle çizilmiş dar açılı köşelerin bilineer modelden sapmasından gelir.")
    R.test("alt piksel doğruluğu", acc)
    if "refine_subpix" not in ns:
        return
    img, P = scene(3, n=12)
    init = np.round(P)
    t1 = bench(lambda: ns["refine_subpix"](img, init), repeat=3)
    c = init.astype(np.float32).reshape(-1, 1, 2)
    t2 = bench(lambda: cv2.cornerSubPix(img, c.copy(), (5, 5), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER, 10, 0)), repeat=3)
    R.table(f"{len(P)} köşe, 10 iterasyon", ["yöntem", "süre"], [["toplu (grid_sample + solve)", fmt_time(t1)], ["cv2.cornerSubPix", fmt_time(t2)]])
