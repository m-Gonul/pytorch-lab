import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        close(need(ns, "power")(20, 0.01, 20), 0.005, what="power")
        close(need(ns, "exponential")(40, 0.01, 20), 1e-4, what="exponential")
        ensure([need(ns, "piecewise")(t, [5, 15], [0.01, 0.005, 0.001]) for t in (0, 5, 14, 15, 99)] == [0.01, 0.005, 0.005, 0.001, 0.001], "piecewise")
        oc = [need(ns, "one_cycle")(t, 100, 0.01, 0.1) for t in range(101)]
        close((oc[0], oc[45], oc[90], oc[100]), (0.01, 0.1, 0.01, 1e-5), atol=1e-9, what="1cycle kilit noktaları")
        ensure(need(ns, "plateau")([5, 4, 4, 4, 4, 3, 3, 3, 3], 1.0, 0.5, 3) == [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.25], "plateau")
        rng = np.random.default_rng(0)
        lam = np.logspace(0, 2, 20)
        T = 300
        rows = []
        res = {}
        for name, sched in (("sabit 0.015", lambda t: 0.015), ("power (s = 30)", lambda t: ns["power"](t, 0.015, 30)), ("üstel (s = 150)", lambda t: ns["exponential"](t, 0.015, 150)),
                            ("piecewise", lambda t: ns["piecewise"](t, [150, 250], [0.015, 0.003, 0.0006])), ("1cycle", lambda t: ns["one_cycle"](t, T, 0.0015, 0.019))):
            x = np.ones(20)
            for t_ in range(T):
                g = lam * x + rng.normal(0, 3, 20)
                x = x - sched(t_) * g
            res[name] = 0.5 * np.sum(lam * x ** 2)
            rows.append([name, f"{res[name]:.3f}"])
        ensure(res["1cycle"] < res["sabit 0.015"] and res["piecewise"] < res["sabit 0.015"], "azalan oran gürültü tabanını düşürür")
        R_.table("Gürültülü kuadratik (κ = 100, gradyan gürültüsü σ = 3), 300 SGD adımı: son kayıp", ["takvim", "kayıp"], rows,
                 note="Sabit oran gürültü tabanında takılır (kayıp ∝ η·σ²). Azalan takvimler önce hızlı ilerleyip sonra gürültüyü ortalar. 1cycle (Smith) ortada büyük oranla 'keşfeder', sonunda oranı birkaç mertebe düşürerek ince ayar yapar — Géron'un önerisi: aynı epoch sayısında çoğu zaman en iyisi.")
    R_.test("öğrenme oranı takvimleri", t)
