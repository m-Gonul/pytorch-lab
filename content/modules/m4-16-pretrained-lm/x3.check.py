import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        B, T, V = 64, 128, 30
        X = rng.integers(4, V, (B, T))
        X[:, 0] = 1
        good = np.full((B, T, V), 0.1 / (V - 1))
        np.put_along_axis(good, X[..., None], 0.9, -1)
        Xc, lab = need(ns, "electra_corrupt")(X, np.random.default_rng(1), good, 4)
        ensure(np.array_equal(lab, Xc != X) and not lab[:, 0].any(), "etiket = değişti mi")
        changed = lab.mean()
        ensure(0.005 < changed < 0.03, f"güçlü üreteç: seçilenlerin çoğu orijinale denk ({changed:.3f})")
        unif = np.full((B, T, V), 1 / V)
        Xu, lu = ns["electra_corrupt"](X, np.random.default_rng(2), unif, 4)
        ensure(0.12 < lu.mean() < 0.16, "zayıf üreteç: seçilenlerin çoğu değişir")
        rows = [["MLM (BERT)", "%15 (seçilenler)", "15%"], ["ELECTRA ayırt edici", "%100 (her token: orijinal mi?)", f"{changed:.1%} / {lu.mean():.1%} değiştirilmiş"],
                ["nedensel LM (GPT)", "%100", "—"]]
        R_.table("Token başına öğrenme sinyali", ["amaç", "kayıp alan konumlar", "pozitif (değiştirilmiş) oranı: iyi / düzgün üreteç"], rows,
                 note="ELECTRA küçük bir MLM üreteciyle maskeli konumları makul token'larla doldurur; ayırt edici her konum için 'değiştirildi mi' sorusunu öğrenir — tüm token'lardan sinyal alır, aynı hesapla BERT'ten belirgin iyidir. Üreteç doğru token'ı örneklediyse o konum 'orijinal' sayılır; üreteç güçlendikçe ayırt edicinin işi zorlaşır (bu yüzden üreteç küçük tutulur).")
    R_.test("ELECTRA", t)
