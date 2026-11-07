import math
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def env():
        f = need(ns, "corridor_step")
        ensure(f(0, True) == (1, -1.0, False) and f(0, False) == (0, -1.0, False), "S")
        ensure(f(1, True) == (0, -1.0, False) and f(1, False) == (2, -1.0, False), "durum 1 ters")
        ensure(f(2, True) == (3, -1.0, True), "hedef")
        v = need(ns, "corridor_value")
        ps = 2 - math.sqrt(2)
        close(v(ps), (2 * ps - 4) / (ps * (1 - ps)), what="J(p*)")
        ensure(v(0.5) < v(0.586) > v(0.7), "p* yerel en büyük")
    R_.test("ortam ve J(p)", env)

    def learn():
        rows, res = [], {}
        for name, kw in [("REINFORCE α = 2⁻¹²", dict(alpha=2 ** -12)), ("REINFORCE α = 2⁻¹³", dict(alpha=2 ** -13)),
                         ("taban çizgili α = 2⁻¹⁰, α_w = 2⁻⁶", dict(alpha=2 ** -10, baseline=True, alpha_w=2 ** -6))]:
            H = [need(ns, "reinforce")(500, seed=s, **kw)[1] for s in range(20)]
            ensure(all(len(h) == 500 for h in H), "geçmiş uzunluğu")
            J = lambda k: sum(ns["corridor_value"](min(max(h[k], 1e-3), 0.999)) for h in H) / 20
            res[name] = [J(k) for k in (24, 49, 99, 199, 499)]
            rows.append([name] + [f"{v:.1f}" for v in res[name]])
        b = res["taban çizgili α = 2⁻¹⁰, α_w = 2⁻⁶"]
        ensure(b[1] > res["REINFORCE α = 2⁻¹²"][1] + 5 and b[-1] > -13, "taban çizgisi daha hızlı ve J* = −11.66'ya yakın")
        R_.table("Kısa koridor: politikanın gerçek değeri J(p), 20 deney ortalaması (J* = −11.66)", ["yöntem", "25", "50", "100", "200", "500. bölüm"], rows,
                 note="Taban çizgisi beklentiyi değiştirmez, varyansı düşürür; bu yüzden daha büyük α kararlı kalır ve öğrenme hızlanır (Sutton Şekil 13.1–13.2).")
    R_.test("REINFORCE ve taban çizgisi", learn)
