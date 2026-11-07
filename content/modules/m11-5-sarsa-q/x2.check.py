from labcheck import need, ensure


def check(ns, R_):
    def t():
        cfg = [("eps", 0.0), ("boltzmann", 0.0), ("greedy", 0.0), ("greedy", 1.0), ("greedy", 2.0), ("greedy", 20.0), ("eps", 2.0)]
        rows, res = [], {}
        for st, q0 in cfg:
            out = [need(ns, "run")(st, q0, 5000, 0.95, 0.2, s) for s in range(5)]
            res[(st, q0)] = out
            rows.append([st, q0, f"{sum(o[0] for o in out) / 5:.4f}", f"{sum(o[1] for o in out)}/5"])
        ensure(sum(o[1] for o in res[("eps", 0.0)]) == 0 and sum(o[1] for o in res[("boltzmann", 0.0)]) == 0, "ε-açgözlü ve Boltzmann tuzağa düşmeli")
        ensure(sum(o[1] for o in res[("greedy", 2.0)]) == 5, "Q0 = 2 ile açgözlü ajan yolu bulmalı")
        ensure(sum(o[0] for o in res[("greedy", 2.0)]) > sum(o[0] for o in res[("greedy", 20.0)]), "çok iyimser Q0 keşfi uzatır")
        R_.table("Zincir, 5000 adım, 5 tohum", ["strateji", "Q0", "ort. ödül/adım", "doğru politika"], rows,
                 note="Sürekli ilerlemek 0.1/adım, yerel döngü 0.05/adım. Q0 = 1 tuzaktaki değere (0.05/(1−γ) = 1) eşit olduğu için yetmez; Q0 = 2 ≥ V* ≈ 1.57 olunca her denenmemiş eylem 'cazip' kalır.")
    R_.test("keşif stratejileri", t)
