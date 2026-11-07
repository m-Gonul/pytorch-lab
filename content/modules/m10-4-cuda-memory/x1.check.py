from labcheck import need, ensure


def check(ns, R_):
    def t():
        f = need(ns, "can_run")
        cases = [
            ([("sm", 8, 0)], (8, 6), "cubin"), ([("sm", 8, 6)], (8, 0), None), ([("sm", 7, 5)], (8, 0), None),
            ([("compute", 7, 5)], (8, 9), "jit"), ([("compute", 9, 0)], (8, 9), None),
            ([("sm", 8, 0), ("compute", 8, 0)], (9, 0), "jit"), ([("sm", 8, 0), ("compute", 8, 0)], (8, 6), "cubin"),
        ]
        rows = []
        for b, cc, want in cases:
            got = f(b, cc)
            ensure(got == want, f"{b} @ {cc}: {want} beklenirdi, {got}")
            rows.append([", ".join(f"{k}_{a}{c}" for k, a, c in b), f"{cc[0]}.{cc[1]}", got or "çalışmaz"])
        R_.table("Gömülü kod ve cihaz hesap yeteneği", ["derleme", "cihaz", "sonuç"], rows,
                 note="Yalnız cubin gömülü bir uygulama yeni nesil GPU'da çalışmaz; PTX de gömülürse sürücü JIT ile derler (ilk çalıştırmada gecikme, önbelleğe alınır). Dağıtım için `-gencode` ile birkaç sm_XY + en yüksek compute_XY gömülür.")
    R_.test("uyumluluk", t)
