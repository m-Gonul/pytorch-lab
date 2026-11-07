import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        res = {}
        for name, f in [("1/n^0.6", lambda n: n ** -0.6), ("1/n", lambda n: 1 / n), ("sabit 0.1", lambda n: 0.1), ("1/n²", lambda n: 1 / n ** 2)]:
            res[name] = np.mean([need(ns, "run")(f, 2000, -3.0, s) for s in range(10)], 0)
        ensure(res["1/n^0.6"][-1] < 0.05, "1/n^0.6 yakınsamalı")
        ensure(res["1/n"][-1] < res["1/n"][99], "1/n yavaş da olsa iyileşmeli")
        ensure(res["1/n²"][-1] > 1.0, "1/n² kötü başlangıçtan kurtulamamalı")
        ensure(res["sabit 0.1"][-1] > res["1/n^0.6"][-1], "sabit α gürültü tabanında kalmalı")
        R_.table("V0 = −3 (gerçek değerler 0.17–0.83), 10 deney", ["α_n", "100. bölüm", "2000. bölüm"], [[k, f"{v[99]:.3f}", f"{v[-1]:.3f}"] for k, v in res.items()],
                 note="Σα = ∞ başlangıç hatasını unutmayı, Σα² < ∞ gürültünün sönmesini sağlar. Koşullar yakınsamayı garanti eder, hızı değil: 1/n TD'de logaritmik hızla unutur.")
    R_.test("Robbins–Monro koşulları", t)
