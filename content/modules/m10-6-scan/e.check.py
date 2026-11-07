import time
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R_):
    src = ns["__source__"]
    R_.test("rec_scan yalnız seviye döngüsü", lambda: (max_loops(src, 1, func="rec_scan"), max_loops(src, 0, func="rec_cumprod"), max_loops(src, 0, func="compact_scan")))
    g = torch.Generator().manual_seed(0)

    def rec():
        rows = []
        for L, lo in ((256, 0.5), (4096, 0.95), (4096, 0.5)):
            a = torch.rand(64, L, generator=g) * (1 - lo) + lo
            b = torch.randn(64, L, generator=g)
            ref = need(ns, "rec_loop")(a.double(), b.double())
            r = []
            for name in ("rec_loop", "rec_scan", "rec_cumprod"):
                out = need(ns, name)(a, b)
                err = float((out.double() - ref).abs().max())
                t = bench(lambda: ns[name](a, b), repeat=2, number=1)
                r.append((name, err, t))
            ensure(r[1][1] < 1e-3, "tarama doğru")
            for name, err, t in r:
                rows.append([f"L = {L}, a ∈ [{lo}, 1)", name, fmt_time(t), "nan" if err != err else f"{err:.1e}"])
        ensure(any(r_[3] in ("nan", "inf") or float(r_[3]) > 1 for r_ in rows if r_[1] == "rec_cumprod"), "kümülatif çarpım uzun dizide bozulmalı")
        R_.table("x_t = a_t x_{t−1} + b_t, 64 dizi (float32)", ["durum", "yöntem", "süre", "max |hata|"], rows,
                 note="Döngü L adım (her biri küçük tensör işlemi), tarama log₂L adım. Kapalı form kümülatif çarpım tek geçiştir ama P_t = Πa küçülünce b/P_t taşar (0.5⁴⁰⁹⁶ = 0): matematiksel olarak doğru, sayısal olarak felaket. Tarama her ara adımda yalnız sınırlı çarpımlar tutar.")
    R_.test("doğrusal özyineleme", rec)

    def comp():
        x = torch.randn(1_000_000, generator=g)
        keep = x > 1.0
        close(need(ns, "compact_scan")(x, keep).numpy(), x[keep].numpy(), what="sıkıştırma")
        R_.table("1 M eleman akış sıkıştırma", ["yöntem", "süre"], [["cumsum + scatter", fmt_time(bench(lambda: ns["compact_scan"](x, keep), repeat=3, number=1))], ["x[keep] (kütüphane)", fmt_time(bench(lambda: x[keep], repeat=3, number=1))]],
                 note="Boolean indekslemenin GPU uygulaması da aynı fikirdir: bayrakların dışlayıcı taraması → çıkış konumları → dağıtım.")
    R_.test("akış sıkıştırma", comp)
