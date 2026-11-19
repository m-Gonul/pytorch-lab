from labcheck import close, need, ensure
import random
import numpy as np


def check(ns, R_):
    def t():
        step = need(ns, "access_step")
        close(step(0, 3, 1, random.Random(0), p=1)[:1], [10], what="dolu havuzda bütün eski işler biter")
        close(step(0, 3, 1, random.Random(0), p=1)[2], 0, what="boş yoksa ödül yok")
        close(step(10, 2, 1, random.Random(0), p=1, release_new=True)[0], 10, what="yeni iş de bitebilir")
        close(step(10, 2, 1, random.Random(0), p=1, release_new=False)[0], 9, what="yeni iş hariç zamanlama")
        close(step(4, 1, 1, random.Random(0), p=0)[::2], [3, 2], what="bırakma yok; tek kabul")
        out = need(ns, "differential_sarsa")(steps=10000, alpha=.02, beta=.005, seed=3)
        Q = np.array(out["Q"])
        close(Q.shape, [11, 4, 2], what="tablo boyutu")
        close(np.sum(out["visits"]), 10000, what="her gerçek adımda bir ziyaret")
        close(Q[0, :, 1], [0] * 4, what="geçersiz kabul güncellenmez")
        close(out["rho"], (.005 / .02) * Q.sum(), atol=1e-9, what="aynı δ ile gain/ağırlık toplamı değişmezi")
        ensure(0 <= out["reward_mean"] <= 8 and np.isfinite(Q).all(), "gerçek ödül ve sonlu ağırlıklar")
        R_.table("10000 gerçek geçiş", ["ortalama ödül", "son rho", "ziyaret"], [[out["reward_mean"], out["rho"], int(np.sum(out["visits"]))]])
    R_.test("Sunucu zamanlaması ve diferansiyel güncelleme", t)
