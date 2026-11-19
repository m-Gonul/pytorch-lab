from labcheck import close, need, ensure
import numpy as np


def check(ns, R_):
    def t():
        gain = need(ns, "gain_step")
        close(gain(10., -18., 2, .1, 0., True), [1., .1], what="ilk normalize hedef; n ölçeği")
        close(gain(-100., 202., 2, .1, 0., True)[0], 1., what="aynı hedefin farklı başlangıç katsayısı")
        close(gain(10., -18., 2, .1, 0., False)[0], 8.2, what="kutudaki düz beta-delta sürümü")
        rho, trace = -17., 0.
        for t in range(1, 21):
            rho, trace = gain(rho, 8*(2-rho), 8, .1, trace, True)
            close(rho, 2., what="sabit ortalama hedef")
            close(trace, 1-.9**t, what="iz kütlesi")
        out = need(ns, "differential_n_sarsa")(steps=2000,n=8,seed=2)
        close(out["updates"], 1993, what="halka tamponu ısınma gecikmesi")
        close(np.sum(out["visits"]), 2000, what="gerçek fiziksel adımlar")
        ensure(np.isfinite(np.asarray(out["Q"])).all() and np.isfinite(out["rho"]), "sonlu sonuçlar")
        R_.table("n ölçeği, ilk hata−18", ["naif iz düzeltmesi", "ortalama-hedef düzeltmesi"], [[10+1*(-18),gain(10.,-18.,2,.1,0.,True)[0]]])
    R_.test("Gain başlangıç katsayısı ve n-adım tamponu", t)
