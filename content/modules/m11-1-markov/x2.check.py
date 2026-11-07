import numpy as np
from labcheck import close, need, ensure


def check(ns, R):
    w = 0.3

    def t():
        X = need(ns, "random_phase")(20000, 60, w, 1)
        Y = need(ns, "random_amplitude")(20000, 60, w, 2)
        Z = need(ns, "random_constant")(2000, 500, 3)
        vX, vY = X.var(0), Y.var(0)
        ensure(vX.std() < 0.02, "rastgele fazlı sürecin varyansı zamanla sabit (WSS)")
        close(vY, np.cos(w * np.arange(60)) ** 2, atol=0.05, what="rastgele genlik: Var = cos²(ωt)")
        cX = np.mean(X[:, 10] * X[:, 15]); cX2 = np.mean(X[:, 30] * X[:, 35])
        close(cX, 0.5 * np.cos(w * 5), atol=0.02, what="kovaryans yalnız τ'ya bağlı"); close(cX2, cX, atol=0.03, what="kayma değişmezliği")
        tz, ez = need(ns, "time_vs_ensemble")(Z)
        tx, ex = ns["time_vs_ensemble"](X[:, :60])
        ensure(tz > 0.9 and ez < 0.1, "sabit süreç: zaman ortalamaları yola göre çok değişir (ergodik değil)")
        R.table("Süreçler", ["süreç", "WSS?", "ergodik (ortalamada)?", "zaman ort. std", "topluluk ort. std"],
                [["cos(ωt + θ)", "evet", "evet", f"{tx:.3f}", f"{ex:.3f}"], ["A·cos(ωt)", "hayır", "—", "—", "—"], ["x_t = A", "evet (katı)", "hayır", f"{tz:.3f}", f"{ez:.3f}"]])
    R.test("durağanlık ve ergodiklik", t)
