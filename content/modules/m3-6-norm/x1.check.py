import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        X = rng.normal(2, 3, (64, 5))
        Y = need(ns, "bn_train")(X)
        close((Y.mean(0), Y.std(0)), (np.zeros(5), np.ones(5)), atol=1e-4, what="batch istatistikleri")
        batches = [rng.normal(1.5, 2, (8, 3)) for _ in range(400)]
        m, v = need(ns, "running_stats")(batches, 0.1)
        close(m, [1.5] * 3, atol=0.4, what="kayan ortalama")
        close(v, [4.0] * 3, atol=1.0, what="yansız kayan varyans")
        bb = [np.array([[0.0], [2.0]])]
        close(ns["running_stats"](bb, 1.0)[1], [2.0], what="N/(N − 1) düzeltmesi (popülasyon varyansı 1 → 2)")
        pool = rng.normal(0, 1, (2000, 16))
        x = rng.normal(0, 1, 16)
        rows = []
        res = {}
        for B in (2, 4, 8, 32, 128):
            res[B] = need(ns, "bn_noise")(x, pool, B, 200, rng)
            rows.append([B, f"{res[B]:.3f}"])
        ensure(res[2] > 0.5 and res[128] < 0.15 and res[4] > res[32], "gürültü batch küçüldükçe artar")
        R_.table("Aynı örneğin BN çıktısı, farklı batch arkadaşlarıyla: denemeler arası std", ["batch boyutu", "çıktı std'si"], rows,
                 note="B = 2'de BN çıktısı yalnız iki değerin farkının işaretidir (±1): örnek hakkında bilgi neredeyse yok olur. Gürültü ~1/√B azalır. Küçük batch'te (tespit, segmentasyon, büyük modeller) BN istatistikleri güvenilmezdir; GroupNorm her örneği kendi kanal grupları üzerinden normalize ettiği için batch'ten bağımsızdır (M3.6 e aşaması).")
    R_.test("BN ve batch boyutu", t)
