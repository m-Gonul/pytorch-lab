from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def mem():
        s = need(ns, "sectors")
        ensure([s([4 * k * t for t in range(32)]) for k in (1, 2, 4, 8, 32)] == [4, 8, 16, 32, 32], "adımlı float okuma")
        ensure(s([4 * t + 16 for t in range(32)]) == 5, "hizasız başlangıç bir sektör fazla")
        ensure(s([0] * 32) == 1, "tüm warp aynı adres")
        ensure(s([8 * t for t in range(32)]) == 8, "float64 ardışık")
        b = need(ns, "bank_degree")
        ensure([b([k * t for t in range(32)]) for k in (1, 2, 3, 4, 16, 32, 33)] == [1, 2, 1, 4, 16, 32, 1], "çakışma dereceleri")
        ensure(b([5] * 32) == 1, "yayın")
        ensure(b([t // 2 for t in range(32)]) == 1, "iki thread aynı kelime: yayın")
    R_.test("sektör ve bank", mem)

    def occ():
        ensure(need(ns, "divergent_paths")([t % 2 for t in range(32)]) == 2 and ns["divergent_paths"]([0] * 32) == 1, "yollar")
        o = need(ns, "occupancy")
        rows = []
        for tpb, regs, smem in ((256, 64, 0), (256, 32, 0), (128, 32, 48 * 1024), (1024, 32, 0), (32, 16, 0), (512, 128, 0)):
            blocks, occ_, why = o(tpb, regs, smem)
            rows.append([tpb, regs, smem, blocks, f"{occ_:.2f}", why])
        ensure(o(256, 64, 0)[:2] == (4, 0.5) and o(256, 64, 0)[2] == "regs", "el örneği")
        ensure(o(32, 16, 0)[2] == "blocks" and o(128, 32, 48 * 1024)[2] == "smem", "sınırlayıcılar")
        R_.table("Doluluk (SM: 2048 thread, 32 blok, 64 K yazmaç, 100 KB paylaşımlı)", ["thread/blok", "yazmaç/thread", "smem/blok", "blok/SM", "doluluk", "sınırlayan"], rows,
                 note="Küçük bloklarda blok sınırı, ağır çekirdeklerde yazmaçlar doluluğu düşürür. Yüksek doluluk gecikme gizlemek için bir araçtır, amaç değil (Volkov ödevi).")
    R_.test("doluluk", occ)
