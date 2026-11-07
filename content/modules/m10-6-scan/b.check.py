import random
from labcheck import no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    add = lambda a, b: a + b
    x = [3, 1, 7, 0, 4, 1, 6, 3]

    def basics():
        inc = [3, 4, 11, 11, 15, 16, 22, 25]
        ensure(need(ns, "inclusive_scan")(x, add) == inc, "kapsayıcı")
        ensure(need(ns, "exclusive_scan")(x, add, 0) == [0] + inc[:-1], "dışlayıcı")
        ensure(need(ns, "kogge_stone")(x, add) == (inc, 3, 17), "Kogge–Stone (sonuç, adım, iş)")
        ensure(need(ns, "brent_kung")(x, add) == (inc, 5, 11), "Brent–Kung (sonuç, adım, iş)")
        ensure(ns["inclusive_scan"](list("tarama"), add)[-1] == "tarama", "değişmeli olmayan işleç")
        ensure(ns["kogge_stone"](list("paralel"), add)[0][-1] == "paralel", "KS sırayı korur")
    R_.test("taramalar", basics)

    def race():
        ok = need(ns, "kogge_stone_inplace")(x, "desc")
        bad = ns["kogge_stone_inplace"](x, "asc")
        ensure(ok == [3, 4, 11, 11, 15, 16, 22, 25] and bad != ok, "azalan sıra doğru, artan sıra yanlış")
        rows = []
        for n in (8, 64, 1024, 65536):
            k = ns["kogge_stone"]([1] * n, add)
            b = ns["brent_kung"]([1] * n, add)
            ensure(k[0] == b[0] == list(range(1, n + 1)), f"n = {n}")
            rows.append([n, n - 1, f"{k[1]} / {k[2]}", f"{b[1]} / {b[2]}", n * (n - 1) // 2])
        rng = random.Random(0)
        y = [rng.randint(-9, 9) for _ in range(10000)]
        ensure(need(ns, "hierarchical_scan")(y, 256, add) == ns["inclusive_scan"](y, add), "hiyerarşik tarama")
        R_.table("Adım / iş (toplama)", ["n", "sıralı iş", "Kogge–Stone", "Brent–Kung", "naif paralel iş"], rows,
                 note=f"Yerinde KS, artan sırada çalışan 'thread'lerle {bad} üretti: thread i, x[i − s]'nin bu adımda güncellenmiş hâlini okudu. Azalan sıra tesadüfen doğru — GPU'da sıra garanti değil; çift tampon ya da oku/senkronize/yaz şart.")
    R_.test("yarış ve karmaşıklık", race)
