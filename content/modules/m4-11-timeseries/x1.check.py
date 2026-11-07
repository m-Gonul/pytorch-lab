import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        close(need(ns, "seasonal_difference")(np.arange(10.) ** 2, 3), (np.arange(3, 10.) ** 2 - np.arange(7.) ** 2), what="mevsimsel fark")
        close(need(ns, "seasonal_naive")(np.arange(10.), 3, 5), [7, 8, 9, 7, 8], what="son mevsimi tekrarla")
        y = np.add.outer(np.zeros(1), np.tile([1., 3, 2, 5, 4, 2, 1], 6))[0] + 0.5 * np.arange(42)
        close(need(ns, "sarima_lite")(y, 7, 1, 7), np.tile([1., 3, 2, 5, 4, 2, 1], 1) + 0.5 * np.arange(42, 49), atol=1e-6, what="trend + mevsim tam tahmin edilir")
        rng = np.random.default_rng(0)
        err = {"naif": [], "mevsimsel naif": [], "SARIMA-benzeri": []}
        for _ in range(60):
            n, h = 140, 14
            t = np.arange(n + h)
            pattern = rng.normal(0, 3, 7)
            e = np.zeros(n + h)
            for i in range(1, n + h):
                e[i] = 0.7 * e[i - 1] + rng.normal(0, 0.5)
            y = 10 + 0.3 * t + pattern[t % 7] + e
            tr, te = y[:n], y[n:]
            err["naif"].append(np.abs(np.full(h, tr[-1]) - te).mean())
            err["mevsimsel naif"].append(np.abs(ns["seasonal_naive"](tr, 7, h) - te).mean())
            err["SARIMA-benzeri"].append(np.abs(ns["sarima_lite"](tr, 7, 2, h) - te).mean())
        m = {k: float(np.mean(v)) for k, v in err.items()}
        ensure(m["SARIMA-benzeri"] < m["mevsimsel naif"] < m["naif"], f"MAE sırası {m}")
        R_.table("Trend + haftalık mevsim + AR(1) gürültü, 60 seri, 14 gün ileri", ["yöntem", "MAE"], [[k, f"{v:.2f}"] for k, v in m.items()],
                 note="Mevsimsel naif (Géron'un yolcu verisindeki 'geçen haftanın aynı günü') naif tahmini rahat geçer ama trendi kaçırır (her hafta 7 × 0.3 geride). (1 − B)(1 − B⁷) farkı hem trendi hem mevsimi siler; kalan AR yapı birkaç gün ileri bilgi verir. Tam SARIMA bu iskeletin üstüne MA ve mevsimsel AR/MA terimleri ekler.")
    R_.test("mevsimsel tahmin", t)
