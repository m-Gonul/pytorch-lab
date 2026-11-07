import numpy as np
import cv2
from skimage import data
from labcheck import close, no_loops, need, ensure


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
    R.test("döngü yok", lambda: no_loops(src))
    img = data.camera()

    def bm():
        prev = img[100:228, 150:278]
        cur = cv2.warpAffine(img, np.float32([[1, 0, -3], [0, 1, 2]]), (512, 512))[100:228, 150:278]
        got = need(ns, "block_match_np")(prev, cur, 8, 5)
        ensure(np.array_equal(got, ref_bm(prev.astype(int), cur.astype(int), 8, 5)), "başvuru (döngülü tam arama) ile farklı")
    R.test("block_match_np = tam arama", bm)

    def lk():
        I0 = cv2.GaussianBlur(img.astype(np.float64), (0, 0), 1.5)
        rows = []
        for u, v in [(0.3, -0.2), (0.6, 0.4)]:
            I1 = cv2.warpAffine(I0, np.float64([[1, 0, u], [0, 1, v]]), (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            uu, vv = need(ns, "lk_flow")(I0, I1)
            Iy, Ix = np.gradient(I0)
            from scipy import ndimage
            f = lambda a: ndimage.uniform_filter(a, 7, mode="nearest")
            a, b, c = f(Ix * Ix), f(Ix * Iy), f(Iy * Iy)
            lmin = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + b * b)
            m = lmin > 20; m[:16] = m[-16:] = False; m[:, :16] = m[:, -16:] = False
            e = np.hypot(uu - u, vv - v)[m]
            rows.append([f"({u}, {v})", f"{np.median(uu[m]):.3f}, {np.median(vv[m]):.3f}", f"{np.median(e):.3f}"])
            ensure(np.median(e) < 0.15, f"medyan uç nokta hatası {np.median(e):.3f}")
        R.table("Tek adımlı LK, dokulu pikseller", ["gerçek (u, v)", "medyan kestirim", "medyan hata"], rows,
                note="Tek adım küçük kaymalarda iyidir; büyük kaymalarda doğrusallaştırma bozulur → iterasyon ve piramit (d).")
    R.test("lk_flow", lk)
