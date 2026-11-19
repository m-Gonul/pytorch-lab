from labcheck import close, need, ensure
import numpy as np


def check(ns, R_):
    def t():
        out = need(ns, "differential_q_demo")(steps=50000,seed=0,eta=.1)
        Q = np.asarray(out["Q"])
        close(np.sum(out["visits"]),50000,what="bütün gerçek adımlar")
        ensure(np.min(out["visits"])>1000,"bütün durum-eylem çiftleri kapsanıyor")
        close(out["rho"], .1*Q.sum(),atol=1e-9,what="aynı hata/adım oranı değişmezi")
        ensure(Q[0,1]>Q[0,0],"en iyi gain politikasında S0→S1")
        close(out["rho"],8/3,atol=.15,what="modelin tam gain değeri, sonlu örnek payıyla")
        ensure(abs(out["behavior_reward"]-out["rho"])>.1,"davranış ödülü ve hedef gain ayrılmalı")
        R_.table("50000 geçiş; farklı hedef/davranış",["hedef gain", "gerçek davranış ortalaması", "tam optimum"],[[out["rho"],out["behavior_reward"],8/3]])
    R_.test("Diferansiyel Q: kapsam ve doğru hedef",t)
