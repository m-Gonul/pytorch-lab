import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def ref_nlm(img, h, tw, sw, sigma):
    H, W = img.shape
    t, s = tw // 2, sw // 2
    P = np.pad(img, t + s, mode="reflect")
    out = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            cy, cx = y + t + s, x + t + s
            pp = P[cy - t:cy + t + 1, cx - t:cx + t + 1]
            num = den = 0.0
            for dy in range(-s, s + 1):
                for dx in range(-s, s + 1):
                    q = P[cy + dy - t:cy + dy + t + 1, cx + dx - t:cx + dx + t + 1]
                    w = np.exp(-max(((pp - q) ** 2).mean() - 2 * sigma ** 2, 0) / h ** 2)
                    num += w * P[cy + dy, cx + dx]; den += w
            out[y, x] = num / den
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız ofset döngüsü", lambda: max_loops(src, 1, func="nlm"))
    rng = np.random.default_rng(0)
    small = np.clip(data.camera()[200:224, 220:244] + rng.normal(0, 15, (24, 24)), 0, 255)
    for h, tw, sw, sg in [(12.0, 3, 7, 0.0), (20.0, 5, 9, 15.0)]:
        def t(h=h, tw=tw, sw=sw, sg=sg):
            got = need(ns, "nlm")(torch.tensor(small, dtype=torch.float64), h, tw, sw, sg).numpy()
            close(got, ref_nlm(small, h, tw, sw, sg), atol=1e-8, what="naif NLM")
        R.test(f"nlm = naif tanım (h = {h}, yama {tw}, arama {sw}, σ = {sg})", t)
    if "nlm" not in ns:
        return
    clean = data.camera()[128:384, 128:384]
    noisy = np.clip(clean + rng.normal(0, 15, clean.shape), 0, 255).astype(np.uint8)
    P = lambda a: cv2.PSNR(clean, np.clip(np.rint(a), 0, 255).astype(np.uint8))
    mine = {h: P(ns["nlm"](torch.tensor(noisy, dtype=torch.float32), h, 7, 21, 0.0).numpy()) for h in (8.0, 10.0, 12.0)}
    cvs = {h: P(cv2.fastNlMeansDenoising(noisy, None, h, 7, 21)) for h in (10, 15, 20)}
    hm, hc = max(mine, key=mine.get), max(cvs, key=cvs.get)
    ensure(mine[hm] > cvs[hc] - 1.0, f"senin en iyin {mine[hm]:.2f}, OpenCV'nin en iyisi {cvs[hc]:.2f}")
    t1 = bench(lambda: ns["nlm"](torch.tensor(noisy, dtype=torch.float32), hm, 7, 21), repeat=1, number=1)
    t2 = bench(lambda: cv2.fastNlMeansDenoising(noisy, None, hc, 7, 21), repeat=3)
    R.table("σ = 15, camera 256², yama 7, arama 21", ["yöntem", "PSNR (dB)", "süre"],
            [["gürültülü", f"{P(noisy):.2f}", "—"], ["Gauss 5 × 5", f"{P(cv2.GaussianBlur(noisy, (5, 5), 0)):.2f}", "—"], ["bilateral (9, 50, 3)", f"{P(cv2.bilateralFilter(noisy, 9, 50, 3)):.2f}", "—"],
             [f"senin NLM (h = {hm:g})", f"{mine[hm]:.2f}", fmt_time(t1)], [f"cv2.fastNlMeansDenoising (h = {hc})", f"{cvs[hc]:.2f}", fmt_time(t2)]],
            note="OpenCV tamsayı aritmetiği ve ağırlık tablolarıyla aynı fikri uygular; h'nin ölçeği farklı olduğu için her biri kendi en iyi h'siyle karşılaştırıldı.")
