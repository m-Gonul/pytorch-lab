from labcheck import need, ensure, fmt_time


def check(ns, R_):
    def t():
        r = need(ns, "time_blocks")(128, 56)
        ensure(set(r) == {"3×3 standart", "3×3 ayrılabilir", "3×3 grup (g = 8)", "1×1"}, "dört blok")
        st, sp = r["3×3 standart"], r["3×3 ayrılabilir"]
        ensure(sp[0] < st[0] / 7, "ayrılabilir MAC ~8× az")
        ensure(sp[2] < st[2], "ayrılabilirin GMAC/s'si (verimi) düşük olmalı")
        R_.table("C = 128, 56×56, batch 8 (CPU)", ["blok", "MAC", "süre", "GMAC/s"], [[k, f"{m / 1e9:.2f} G", fmt_time(t), f"{v:.1f}"] for k, (m, t, v) in r.items()],
                 note=f"Ayrılabilir blok {st[0] / sp[0]:.1f}× az MAC yapar ama süre yalnız {st[1] / sp[1]:.1f}× kısalır: derinlemesine kısım kanal başına küçük, bellek-bağımlı bir işlemdir. Mobil donanım ve özel çekirdekler bu farkı kapatmaya çalışır.")
    R_.test("FLOP ve süre", t)
