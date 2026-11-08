import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        close(need(ns, "ema")(np.array([1.0, 1.0, 1.0]), 0.9), [0.1, 0.19, 0.271], what="EMA")
        close(ns["ema"](np.array([2.0, 2.0, 2.0]), 0.9, True), [2.0, 2.0, 2.0], what="bias düzeltmesi sabit sinyali korur")
        rng = np.random.default_rng(0)
        T = 2000
        trend = np.sin(np.arange(T) / 150)
        g = trend + rng.normal(0, 1.0, T)
        rows = []
        for b in (0.01, 0.5, 0.9, 0.99):
            f = ns["ema"](g, b, True)
            noise = np.std(f[300:] - trend[300:])
            lag = int(np.argmax([np.corrcoef(f[300 + k:1800 + k], trend[300:1800])[0, 1] for k in range(0, 120)]))
            rows.append([b, f"{1 / (1 - b):.0f}", f"{noise:.3f}", lag])
        ensure(float(rows[0][2]) > 0.9 and float(rows[2][2]) < 0.4, "β büyüdükçe gürültü azalır")
        ensure(rows[3][3] > rows[1][3], "β → 1: gecikme artar")
        R_.table("Dalgalı gradyan = yavaş trend + N(0, 1) gürültü (ders alıştırması 3)", ["unutma faktörü β", "etkin pencere 1/(1 − β)", "trend etrafındaki hata std", "gecikme (adım)"], rows,
                 note="β = 0.01'de filtre neredeyse hiçbir şeyi hatırlamaz — sinyal ham gradyan kadar gürültülü. β = 0.99'da son ~100 gradyanın ortalaması: gürültü kaybolur ama trend değişimleri gecikmeli izlenir. 'Unutma faktörü' adı bundan: β, eski gradyanların her adımda ne kadarının hatırlandığıdır.")
        A = np.diag([1.0, 100.0])
        x0 = np.array([1.0, 1.0])
        k = 100.0
        lr = 4 / (1 + np.sqrt(k)) ** 2
        beta = ((np.sqrt(k) - 1) / (np.sqrt(k) + 1)) ** 2
        hb = need(ns, "heavy_ball")(A, x0, lr, beta, 150)
        ne = need(ns, "nesterov")(A, x0, 1 / 100, (np.sqrt(k) - 1) / (np.sqrt(k) + 1), 150)
        sgd = ns["heavy_ball"](A, x0, 2 / 101, 0.0, 150)
        ensure(hb[-1] < 1e-8 * hb[0] and sgd[-1] > 1e-3 * sgd[0] and ne[-1] < 1e-4 * ne[0], "momentum koşul sayısının karekökü kadar hızlandırır")
        R_.note(f"κ = 100'lük kuadratikte 150 adım sonra kayıp: SGD (en iyi η) {sgd[-1]:.2e}, ağır top (Polyak, en iyi η, β) {hb[-1]:.2e}, Nesterov (η = 1/L, β = (√κ − 1)/(√κ + 1)) {ne[-1]:.2e}. Yakınsama oranı SGD'de ≈ 1 − 2/κ, momentumla ≈ 1 − 2/√κ.")
    R_.test("EMA, ağır top, Nesterov", t)
