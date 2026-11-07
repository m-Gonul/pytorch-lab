from labcheck import need, ensure


def check(ns, R_):
    def t():
        n = 32
        ensure(sum(1 for _ in need(ns, "trace")("ijk", 4)) == 4 * 4 * (2 * 4 + 1), "ijk iz uzunluğu")
        ensure(sorted(set(need(ns, "trace")("tiled", 8, 4))) == sorted(set(ns["trace"]("ijk", 8))), "aynı adres kümesi")
        rows = []
        res = {}
        for order, T in (("ijk", 4), ("ikj", 4), ("tiled", 4), ("tiled", 8)):
            m = need(ns, "misses")(order, n, 8, 32, T)
            res[(order, T)] = m
            rows.append([order if order != "tiled" else f"karo T = {T}", m, f"{m / (3 * n * n / 8):.1f}×"])
        ensure(res[("ijk", 4)] > res[("ikj", 4)] > res[("tiled", 8)], "ijk > ikj > karo")
        R_.table(f"n = {n}, satır 8 kelime, önbellek 32 satır (LRU)", ["sıra", "kaçırma", "zorunlu kaçırmaya oranı"], rows,
                 note="Zorunlu (soğuk) kaçırma = 3n²/8 satır. ijk B'yi sütun boyunca okur: her erişim yeni satır. ikj ardışık okur ama B'nin tamamı önbelleğe sığmaz. Karo, üç küçük bloğu önbellekte tutup her elemanı T kez yeniden kullanır; T, 3T² ≤ önbellek olacak şekilde seçilir.")
    R_.test("önbellek simülasyonu", t)
