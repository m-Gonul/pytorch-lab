import numpy as np
from labcheck import need, ensure


def check(ns, R):
    def t():
        f = need(ns, "bandit_sweep")
        configs = [("epsilon", 0), ("constant", .1), ("ucb", 2), ("optimistic", 5), ("gradient", .1)]
        zero = f(8, 100, configs, 7, drift=0, noise=0)
        ensure(np.allclose(zero, 0), "Sıfır ödüllü ortamda her yöntem ve ölçüm sıfır olmalı.")
        deterministic = f(64, 400, configs, 11, drift=0, noise=0, means=[0, 1])
        ensure(deterministic.shape == (5,), "Her ayar için tek ortalama dönmeli.")
        ensure(np.all((deterministic >= -1e-12) & (deterministic <= 1 + 1e-12)), "Ödüller 0/1 iken ortalama [0,1] içinde olmalı.")
        ensure(deterministic[0] > .99 and deterministic[1] > .85 and deterministic[2] > .9, "Gürültüsüz iki kolu öğrenme veya ε/UCB kuralı yanlış.")
        moving = f(12, 600, configs, 8)
        ensure(np.isfinite(moving).all(), "Değişen ortamda sonlu ortalamalar gerekli.")
        ensure(np.array_equal(moving, f(12, 600, configs, 8)), "Aynı tohum aynı deneyi vermeli.")
        R.table("Kısa bağımsız denetim; kaynak uzun deneyi değildir", ["yöntem", "gürültüsüz ortalama", "değişen ortam ortalaması"],
                [[label, float(a), float(b)] for label, a, b in zip(["örnek ortalaması", "sabit α", "UCB", "iyimser başlangıç", "gradyan"], deterministic, moving)],
                note="Kaynak 2.11 için 200000 adım ve son 100000 ortalaması ayrıca çalıştırılmalı.")
    R.test("uzun tarama API'si ve ödül semantiği", t)
