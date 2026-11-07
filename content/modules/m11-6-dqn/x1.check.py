from labcheck import need, ensure


def check(ns, R_):
    def t():
        p, v, d = need(ns, "mc_step")(-0.5, 0.0, 2)
        ensure(abs(v - (0.001 - 0.0025 * __import__("math").cos(-1.5))) < 1e-12 and not d, "fizik")
        ensure(ns["mc_step"](-1.19, -0.07, 0)[:2] == (-1.2, 0.0), "sol duvar")
        T = need(ns, "tiles")(-0.5, 0.01, 1)
        ensure(len(T) == 8 and len(set(T)) == 8 and all(i % 3 == 1 for i in T), "8 farklı karo, eylem kodlu")
        res = [need(ns, "sarsa_mc")(100, 0.5, s)[1] for s in range(3)]
        first = sum(r[0] for r in res) / 3
        last = sum(sum(r[-10:]) / 10 for r in res) / 3
        ensure(first > 400 and last < 250, f"bölüm uzunluğu {first:.0f} → {last:.0f}")
        R_.table("Dağ arabası, yarı-gradyan SARSA, 8 karo katmanı, α = 0.5/8 (3 tohum)", ["bölüm", "ort. adım"],
                 [[k + 1, f"{sum(r[k] for r in res) / 3:.0f}"] for k in (0, 1, 4, 9, 24, 49, 99)],
                 note="Araba doğrudan tırmanamaz: önce geri gidip ivme kazanması gerekir. İyimser başlangıç (w = 0) keşfi tek başına sağlar (Sutton Şekil 10.2).")
    R_.test("dağ arabası", t)
