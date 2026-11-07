import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        close(need(ns, "padding_waste")([3, 1, 2, 2], [np.array([0, 1]), np.array([2, 3])]), 1 - 8 / 10, what="israf tanımı")
        rng = np.random.default_rng(0)
        L = np.clip(rng.lognormal(2.7, 0.6, 20000).astype(int), 2, 200)
        rb = need(ns, "random_batches")(len(L), 64, rng)
        bb = need(ns, "bucket_batches")(L, 64, rng, 50)
        ensure(sorted(np.concatenate(bb).tolist()) == list(range(len(L))), "her örnek tam bir kez")
        wr, wb = ns["padding_waste"](L, rb), ns["padding_waste"](L, bb)
        ensure(wb < wr / 3, f"kovalama israfı azaltır ({wr:.2f} → {wb:.2f})")
        rows = [["rastgele batch'ler", f"{wr:.1%}"]]
        for ch in (1, 5, 50, 1000):
            rows.append([f"kovalama, parça = {ch} batch", f"{ns['padding_waste'](L, ns['bucket_batches'](L, 64, rng, ch)):.1%}"])
        R_.table("20 000 cümle (log-normal uzunluk, ort. ≈ 18 token), batch 64", ["yöntem", "dolgu token oranı"], rows,
                 note="Dolgu token'ları hem bellek hem hesap harcar (RNN'de her adım, Transformer'da karesel). Kovalama benzer uzunlukları birlikte koyar; parça büyüdükçe israf azalır ama batch'lerin içeriği daha az rastgele olur (aynı batch'te hep benzer uzunluklar) — parça boyutu bu ödünleşimi ayarlar.")
    R_.test("kovalama", t)
