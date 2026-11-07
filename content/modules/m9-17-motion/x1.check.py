import numpy as np
from skimage import data
from labcheck import close, need, ensure


def video(T=9, sigma=12, seed=0):
    rng = np.random.default_rng(seed)
    bg = data.camera()[100:228, 100:228].astype(np.float64)
    obj = data.coins()[50:90, 50:90].astype(np.float64)
    clean = []
    for t in range(T):
        f = bg.copy(); f[40:80, 10 + 8 * t:50 + 8 * t] = obj
        clean.append(f)
    clean = np.stack(clean)
    return clean, clean + rng.normal(0, sigma, clean.shape)


def check(ns, R):
    clean, noisy = video()

    def tm():
        got = need(ns, "temporal_mean")(noisy, 2)
        close(got[4], noisy[2:7].mean(0), what="orta kare"); close(got[0], noisy[0:3].mean(0), what="uç kare")
    R.test("temporal_mean", tm)

    def cmp():
        P = lambda a, reg=(slice(None), slice(None)): np.mean([10 * np.log10(255 ** 2 / ((a[t][reg] - clean[t][reg]) ** 2).mean()) for t in range(2, 7)])
        mov = (slice(40, 80), slice(None))
        sta = (slice(90, 128), slice(None))
        rows = [[n, f"{P(a):.2f}", f"{P(a, mov):.2f}", f"{P(a, sta):.2f}"] for n, a in [("gürültülü", noisy), ("uzaysal Gauss", need(ns, "spatial")(noisy)), ("zamansal ortalama (5 kare)", ns["temporal_mean"](noisy, 2)), ("hareket kapılı ortalama", need(ns, "gated_mean")(noisy, 2, 12))]]
        d = {r[0]: (float(r[1]), float(r[2]), float(r[3])) for r in rows}
        ensure(d["hareket kapılı ortalama"][1] > d["zamansal ortalama (5 kare)"][1] + 2, "kapı hareketli bölgede hayaleti önlemeli")
        ensure(d["zamansal ortalama (5 kare)"][2] > d["gürültülü"][2] + 5, "zamansal ortalama durağan bölgede PSNR'ı belirgin artırmalı")
        R.table("σ = 12 gürültü, nesne kare başına 8 px kayıyor", ["yöntem", "PSNR (tüm kare)", "PSNR (hareketli şerit)", "PSNR (durağan bölge)"], rows,
                note="Düz zamansal ortalama durağan bölgede √5 kat gürültü azaltır ama hareketli nesneyi hayalete çevirir; kapı ikisini birden yakalar.")
    R.test("karşılaştırma", cmp)
