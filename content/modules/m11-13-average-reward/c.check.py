from labcheck import close, need, ensure
import numpy as np


def check(ns, R_):
    def t():
        P = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        rho, h = need(ns, "average_evaluate")(P, [0., 0., 1.])
        close(rho, 1/3, what="üç halkada gain")
        close(h, [0, 1/3, 2/3], what="hA=0 referanslı Poisson çözümü")
        for include in (True, False):
            D = need(ns, "access_kernel")(.06, include)
            close(D.sum(axis=-1), np.ones((11, 2)), what="sıfır sunucuda satırı iki kez toplama hatası yok")
            close(D[0, 0], D[0, 1], what="zorunlu red geçişi")
        results = []
        for include, eps, expected in [(True, 0., 2.747641950572482), (True, .1, 2.6468943683103894), (False, 0., 2.6731922050681827)]:
            out = need(ns, "optimal_access")(.06, eps, include)
            close(out["rho"], expected, atol=1e-10, what="bağımsız44durum modelinden gain")
            close(out["r"] + out["P"] @ out["h"] - out["h"], np.full(11, out["rho"]), atol=1e-10, what="Poisson kalanı")
            ensure(np.all(out["accept"][0] == 0), "boş0iken red")
            results.append([include, eps, out["rho"]])
        R_.table("Model oracle'ı: kaynakta eksik zamanlama açık", ["yeni iş de bitebilir", "ε", "gain"], results)
    R_.test("Poisson normalizasyonu ve bağımsız access-control referansı", t)
