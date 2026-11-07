import numpy as np
import cv2
from skimage import data
from labcheck import no_imports, need, ensure


def ref_bm(prev, cur, bs, R):
    H, W = cur.shape
    out = np.zeros((H // bs, W // bs, 2), int)
    for i in range(H // bs):
        for j in range(W // bs):
            c = cur[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs].astype(int)
            best = None
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    y, x = i * bs + dy, j * bs + dx
                    if 0 <= y <= H - bs and 0 <= x <= W - bs:
                        s = np.abs(c - prev[y:y + bs, x:x + bs]).sum()
                        if best is None or s < best[0]:
                            best = (s, dy, dx)
            out[i, j] = best[1:]
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/cv2 kullanılmıyor", lambda: no_imports(src, {"numpy", "cv2", "scipy", "skimage"}))
    prev = data.camera()[200:248, 200:256].astype(int)
    M = np.float32([[1, 0, 2], [0, 1, -1]])
    cur = cv2.warpAffine(data.camera(), M, (512, 512))[200:248, 200:256].astype(int)

    def bm():
        got = np.array(need(ns, "block_match")(prev.tolist(), cur.tolist(), 8, 3))
        want = ref_bm(prev, cur, 8, 3)
        ensure(got.shape == want.shape and np.array_equal(got, want), "başvuru ile farklı")
        inner = got[1:-1, 1:-1].reshape(-1, 2)
        ensure((inner == [1, -2]).all(1).mean() > 0.8, "iç blokların çoğu (dy, dx) = (1, −2) olmalı")
    R.test("block_match (tam arama)", bm)

    def mc():
        mv = need(ns, "block_match")(prev.tolist(), cur.tolist(), 8, 3)
        pred = np.array(need(ns, "compensate")(prev.tolist(), mv, 8))
        P = lambda a: 10 * np.log10(255 ** 2 / ((a - cur) ** 2).mean())
        ensure(P(pred) > P(prev) + 5, "telafi, önceki kareyi kopyalamaktan çok daha iyi olmalı")
        return f"önceki kare PSNR {P(prev):.1f} dB → hareket telafili {P(pred):.1f} dB"
    R.test("compensate", mc)
