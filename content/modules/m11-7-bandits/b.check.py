from labcheck import no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t():
        probs = [0.9, 0.8, 0.7, 0.5, 0.3]
        rows, res = [], {}
        for name, pol, kw in [("açgözlü", "eps", {"eps": 0.0}), ("ε = 0.1", "eps", {}), ("ε_t = min(1, 5/t)", "eps_decay", {}), ("UCB1", "ucb", {}), ("Thompson", "ts", {})]:
            runs = [need(ns, "run")(pol, probs, 4000, s, **kw) for s in range(20)]
            ensure(all(len(r) == 4000 for r in runs), "uzunluk")
            r1 = sum(r[999] for r in runs) / 20
            r2 = sum(r[1999] for r in runs) / 20
            r4 = sum(r[-1] for r in runs) / 20
            res[name] = (r1, r4, max(r[-1] for r in runs), (r4 - r2) / max(r2 - r1, 1e-9))
            rows.append([name, f"{r1:.1f}", f"{r4:.1f}", f"{res[name][3]:.2f}", f"{res[name][2]:.0f}"])
        ensure(res["Thompson"][1] < min(v[1] for k, v in res.items() if k != "Thompson"), "Thompson en düşük pişmanlık")
        ensure(res["ε = 0.1"][3] > 1.7 and res["UCB1"][3] < 1.4, "sabit ε doğrusal, UCB1 logaritmik büyümeli")
        ensure(res["açgözlü"][2] > 1000, "açgözlü bazı deneylerde kötü kola takılmalı")
        R_.table("Kollar 0.9, 0.8, 0.7, 0.5, 0.3 — 20 deney ortalama pişmanlık", ["yöntem", "t = 1000", "t = 4000", "artış oranı", "en kötü deney"], rows,
                 note="Artış oranı = [L(4000) − L(2000)] / [L(2000) − L(1000)]: doğrusal pişmanlıkta ≈ 2, logaritmikte ≈ ln 2 / ln 2 = 1. UCB1'in √(2 ln t/n) sınırı muhafazakârdır: başta çok keşfeder, uzun vadede ε'u geçer.")
    R_.test("beş algoritma", t)
