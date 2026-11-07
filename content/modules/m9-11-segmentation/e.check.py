import math
import numpy as np
from skimage import data, color
from labcheck import need, ensure, bench, fmt_time


def init_centers(L, S):
    H, W, _ = L.shape
    P = np.pad(L, ((1, 1), (1, 1), (0, 0)), mode="edge")
    G = ((P[1:-1, 2:] - P[1:-1, :-2]) ** 2).sum(-1) + ((P[2:, 1:-1] - P[:-2, 1:-1]) ** 2).sum(-1)
    cs = []
    for y in range(S // 2, H, S):
        for x in range(S // 2, W, S):
            best = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and (best is None or G[yy, xx] < best[0]):
                        best = (G[yy, xx], yy, xx)
            cs.append([*L[best[1], best[2]], best[1], best[2]])
    return np.array(cs, np.float64)


def slic_ref(L, K, c=10.0, iters=10):
    H, W, _ = L.shape
    S = int(round(math.sqrt(H * W / K)))
    C = init_centers(L, S)
    lab = -np.ones((H, W), np.int64)
    yy, xx = np.mgrid[:H, :W]
    F = np.concatenate([L, yy[..., None], xx[..., None]], -1).reshape(-1, 5)
    for _ in range(iters):
        d = np.full((H, W), np.inf)
        for k, (l, a, b, cy, cx) in enumerate(C):
            y0, y1 = max(0, math.ceil(cy - S)), min(H - 1, math.floor(cy + S))
            x0, x1 = max(0, math.ceil(cx - S)), min(W - 1, math.floor(cx + S))
            win = L[y0:y1 + 1, x0:x1 + 1]
            D = np.sqrt(((win - [l, a, b]) ** 2).sum(-1) / c ** 2 + ((yy[y0:y1 + 1, x0:x1 + 1] - cy) ** 2 + (xx[y0:y1 + 1, x0:x1 + 1] - cx) ** 2) / S ** 2)
            m = D < d[y0:y1 + 1, x0:x1 + 1]
            d[y0:y1 + 1, x0:x1 + 1][m] = D[m]
            lab[y0:y1 + 1, x0:x1 + 1][m] = k
        for k in range(len(C)):
            sel = lab.ravel() == k
            if sel.any():
                C[k] = F[sel].mean(0)
    return lab


def compactness(lab):
    vals = []
    for k in np.unique(lab):
        m = lab == k
        per = (m[:, 1:] != m[:, :-1]).sum() + (m[1:] != m[:-1]).sum()
        vals.append(4 * np.pi * m.sum() / max(per, 1) ** 2)
    return float(np.mean(vals))


def check(ns, R):
    L = color.rgb2lab(data.astronaut()[:160, 120:300])
    for K, c in [(60, 10.0), (150, 5.0), (40, 30.0)]:
        def t(K=K, c=c):
            got = need(ns, "slic")(L, K, c)
            want = slic_ref(L, K, c)
            ensure(got.shape == want.shape and np.array_equal(got, want), f"{int((got != want).sum())} piksel başvurudan farklı")
        R.test(f"slic = pencereli başvuru (K = {K}, c = {c})", t)
    if "slic" not in ns:
        return
    rows = []
    for c in (2.0, 10.0, 40.0):
        lab = ns["slic"](L, 100, c)
        rgb = data.astronaut()[:160, 120:300].astype(float)
        var = float(np.mean([rgb[lab == k].var(0).sum() for k in np.unique(lab)]))
        rows.append([c, len(np.unique(lab)), f"{compactness(lab):.3f}", f"{var:.0f}"])
    ensure(float(rows[2][2]) > float(rows[0][2]), "c büyüdükçe süper pikseller kompaktlaşmalı")
    R.table("c'nin etkisi (K = 100)", ["c", "süper piksel", "ort. kompaktlık 4πA/P²", "ort. renk varyansı"], rows)
    t1 = bench(lambda: slic_ref(L, 100), repeat=1, number=1)
    t2 = bench(lambda: ns["slic"](L, 100), repeat=1, number=1)
    R.table("160 × 180, K = 100, 10 iterasyon", ["yöntem", "süre"], [["merkez başına pencere döngüsü (NumPy)", fmt_time(t1)], ["yoğun P × K maske (torch)", fmt_time(t2)]],
            note="Yoğun sürüm her piksel–merkez çiftini hesaplar (O(PK)); pencereli sürüm yalnız ~4S² pikseli (O(P)). GPU'da yoğun sürüm paralellikle kazanır; CPU'da pencere kazanır.")
