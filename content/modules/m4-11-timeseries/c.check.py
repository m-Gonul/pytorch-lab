import numpy as np
from labcheck import close, max_loops, need, ensure


def _ar(phi, n, seed, sigma=1.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n + 500)
    y = np.zeros(n + 500)
    for t in range(len(y)):
        y[t] = e[t] + sum(p * y[t - 1 - i] for i, p in enumerate(phi) if t - 1 - i >= 0)
    return y[500:]


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüler", lambda: ([max_loops(src, 0, func=f) for f in ("acf", "lag_matrix", "fit_ar_ls", "fit_ar_yw")], max_loops(src, 1, func="pacf"), max_loops(src, 1, func="forecast_ar")))
    y = _ar([0.6, 0.3], 40000, 0)

    def t_fit():
        X, t = need(ns, "lag_matrix")(np.arange(6.), 2)
        ensure(X.tolist() == [[1, 0], [2, 1], [3, 2], [4, 3]] and t.tolist() == [2, 3, 4, 5], "gecikme matrisi")
        z = np.random.default_rng(1).normal(size=50)
        close(need(ns, "acf")(z, 3)[1], np.sum((z[:-1] - z.mean()) * (z[1:] - z.mean())) / np.sum((z - z.mean()) ** 2), what="acf tanımı")
        r = ns["acf"](y, 3)
        close(r[1:3], [0.6 / 0.7, 0.6 * 0.6 / 0.7 + 0.3], atol=0.02, what="örnek ACF ≈ kuram")
        close(need(ns, "fit_ar_ls")(y, 2), [0.6, 0.3], atol=0.02, what="LS")
        close(need(ns, "fit_ar_yw")(y, 2), [0.6, 0.3], atol=0.02, what="Yule–Walker")
        f = need(ns, "forecast_ar")(np.array([1.0, 2.0]), np.array([0.6, 0.3]), 3)
        close(f, [1.5, 0.6 * 1.5 + 0.3 * 2, 0.6 * (0.6 * 1.5 + 0.6) + 0.3 * 1.5], what="yinelemeli tahmin")
    R_.test("AR tahmini", t_fit)

    def t_id():
        y1 = _ar([0.7], 20000, 2)
        y3 = _ar([0.5, -0.3, 0.3], 20000, 3)
        rows = []
        for name, s in (("AR(1) φ = 0.7", y1), ("AR(2) (0.6, 0.3)", y), ("AR(3) (0.5, −0.3, 0.3)", y3)):
            a = ns["acf"](s, 6)
            p = need(ns, "pacf")(s, 6)
            rows.append([name, " ".join(f"{v:+.2f}" for v in a[1:]), " ".join(f"{v:+.2f}" for v in p[1:])])
        p1, p3 = ns["pacf"](y1, 5), ns["pacf"](y3, 5)
        ensure(abs(p1[1] - 0.7) < 0.03 and np.all(np.abs(p1[2:]) < 0.03), "AR(1) PACF 1'den sonra kesilir")
        ensure(abs(p3[3] - 0.3) < 0.03 and np.all(np.abs(p3[4:]) < 0.03), "AR(3) PACF 3'ten sonra kesilir")
        R_.table("Box–Jenkins model belirleme: ACF söner, PACF p'de kesilir", ["süreç", "ACF (gecikme 1–6)", "PACF (gecikme 1–6)"], rows,
                 note="AR(p) sürecinin ACF'si üstel/sönümlü salınımla söner, PACF'si p'den sonra sıfırdır (gecikme p'deki PACF = φ_p). MA(q) için tersi geçerlidir: ACF q'dan sonra kesilir. Gürültü bandı ≈ ±2/√n.")
    R_.test("ACF / PACF", t_id)
