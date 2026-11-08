import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        C, d = 10, 16
        centers = rng.normal(0, 4, (C, d))

        def probs(F):
            D2 = ((F[:, None] - centers[None]) ** 2).sum(-1)
            Z = -D2 / 2
            Z -= Z.max(1, keepdims=True)
            P = np.exp(Z)
            return P / P.sum(1, keepdims=True)
        real = centers[rng.integers(0, C, 5000)] + rng.normal(size=(5000, d))
        cases = {"gerçek dağılım": real,
                 "sınıf başına tek 'mükemmel' örnek (merkezler)": centers[rng.integers(0, C, 5000)],
                 "sınıf başına tek örnek, merkezden uzakta": (centers + rng.normal(size=(C, d)))[rng.integers(0, C, 5000)],
                 "yarı modlar": centers[rng.integers(0, C // 2, 5000)] + rng.normal(size=(5000, d)),
                 "sınıflar arası bulanık (merkezlerin karışımı)": 0.5 * (centers[rng.integers(0, C, 5000)] + centers[rng.integers(0, C, 5000)]) + rng.normal(size=(5000, d))}
        rows, r = [], {}
        for name, F in cases.items():
            r[name] = (need(ns, "inception_score")(probs(F)), need(ns, "fid")(real, F))
            rows.append([name, f"{r[name][0]:.2f}", f"{r[name][1]:.1f}"])
        ens = r["sınıf başına tek 'mükemmel' örnek (merkezler)"]
        ensure(ens[0] >= r["gerçek dağılım"][0] - 0.05 and ens[1] > 5, "IS kandırılır, FID kandırılmaz")
        ensure(r["yarı modlar"][0] < 6 and r["sınıflar arası bulanık (merkezlerin karışımı)"][0] < r["gerçek dağılım"][0], "IS çeşitlilik ve netlik kaybını görür")
        R_.table("10 sınıflı sentetik öznitelik uzayı (Inception vekili: merkezlere uzaklık softmax'ı)", ["üretilen", "IS (≤ 10)", "FID"], rows,
                 note="IS yalnız sınıflandırıcının 'emin mi' ve 'sınıflar dengeli mi' sorularına bakar: her sınıftan tek bir mükemmel görüntüyü sonsuza kadar tekrarlayan bir üretici en yüksek IS'i alır (sınıf içi çeşitlilik sıfır). FID gerçek verinin kovaryansıyla karşılaştırdığı için bunu cezalandırır (Barratt & Sharma 2018).")
    R_.test("IS vs FID", t)
