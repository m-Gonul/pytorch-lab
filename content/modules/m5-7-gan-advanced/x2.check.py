import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        d = 32
        A = rng.normal(size=(d, d)) / np.sqrt(d)
        real = rng.normal(size=(50000, d)) @ A.T
        good = lambda n, r: r.normal(size=(n, d)) @ A.T
        bad = lambda n, r: 1.15 * r.normal(size=(n, d)) @ A.T + 0.05
        Ns = np.array([500, 1000, 2000, 5000, 10000])
        fg = need(ns, "fid_curve")(real, good, Ns, 1)
        fb = ns["fid_curve"](real, bad, Ns, 2)
        ig, ib = need(ns, "fid_inf")(Ns, fg), ns["fid_inf"](Ns, fb)
        ensure(fg[0] > 2 * fg[-1] and abs(ig) < 0.05, f"yanlılık ~1/N, FID∞ ≈ 0 ({ig:.3f})")
        S = A @ A.T
        true_bad = np.sum(np.full(d, 0.05) ** 2) + np.trace(S) * (1 + 1.15 ** 2 - 2 * 1.15)
        close(ib, true_bad, atol=0.05, what="FID∞ gerçek değere yakın")
        rows = [[int(n), f"{a:.3f}", f"{b:.3f}"] for n, a, b in zip(Ns, fg, fb)] + [["∞ (uydurma)", f"{ig:.3f}", f"{ib:.3f}"]]
        R_.table("32 boyutlu Gauss öznitelikler: örnek sayısı N'ye göre FID", ["N", "doğru üretici", "kusurlu üretici"], rows,
                 note=f"Gerçek FID 0 olan üretici bile N = 500'de {fg[0]:.2f} alır; yanlılık ~1/N azalır ve modelden modele değişir — farklı N ile hesaplanmış FID'ler karşılaştırılamaz. Chong & Forsyth: FID'i birkaç N'de hesaplayıp 1/N'ye doğrusal uydurmak (FID∞) yansız bir tahmin verir (kusurlu üreticinin gerçek değeri {true_bad:.3f}).")
    R_.test("FID yanlılığı", t)
