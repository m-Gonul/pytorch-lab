import numpy as np
from labcheck import max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız adım döngüsü", lambda: (max_loops(src, 1, func="testbed"), max_loops(src, 1, func="gradient_bandit")))

    def tb():
        rows, res = [], {}
        for name, kw in [("ε = 0 (açgözlü)", dict(eps=0.0)), ("ε = 0.01", dict(eps=0.01)), ("ε = 0.1", dict(eps=0.1)),
                         ("iyimser Q0 = 5, ε = 0, α = 0.1", dict(eps=0.0, Q0=5.0, alpha=0.1)), ("gerçekçi Q0 = 0, ε = 0.1, α = 0.1", dict(eps=0.1, alpha=0.1))]:
            r, o = need(ns, "testbed")(2000, 1000, seed=1, **kw)
            res[name] = (r[-100:].mean(), o[-100:].mean(), o[:50].mean())
            rows.append([name, f"{res[name][0]:.2f}", f"{100 * res[name][1]:.0f}%", f"{100 * res[name][2]:.0f}%"])
        ensure(res["ε = 0.1"][0] > res["ε = 0.01"][0] > res["ε = 0 (açgözlü)"][0], "Sutton Şekil 2.2 sıralaması (1000 adımda)")
        ensure(res["iyimser Q0 = 5, ε = 0, α = 0.1"][1] > res["gerçekçi Q0 = 0, ε = 0.1, α = 0.1"][1] and res["iyimser Q0 = 5, ε = 0, α = 0.1"][2] < res["gerçekçi Q0 = 0, ε = 0.1, α = 0.1"][2], "iyimser başlangıç: başta kötü, sonra iyi")
        for name, kw in [("gradyan α = 0.1, taban çizgili", dict(alpha=0.1)), ("gradyan α = 0.1, taban çizgisiz", dict(alpha=0.1, baseline=False))]:
            r, o = need(ns, "gradient_bandit")(2000, 1000, seed=1, **kw)
            res[name] = (r[-100:].mean(), o[-100:].mean(), o[:50].mean())
            rows.append([name + " (q* ortalaması 4)", f"{res[name][0]:.2f}", f"{100 * res[name][1]:.0f}%", f"{100 * res[name][2]:.0f}%"])
        ensure(res["gradyan α = 0.1, taban çizgili"][1] > res["gradyan α = 0.1, taban çizgisiz"][1] + 0.15, "taban çizgisi önemli (Şekil 2.5)")
        R_.table("10 kollu test yatağı, 2000 deney × 1000 adım", ["yöntem", "son 100 adım ort. ödül", "son 100 adım optimal", "ilk 50 adım optimal"], rows,
                 note="Taban çizgisi olmadan bütün ödüller ≈ 4 pozitif olduğu için seçilen her eylemin tercihi artar; varyans büyür ve öğrenme yavaşlar.")
    R_.test("test yatağı ve gradyan haydut", tb)
