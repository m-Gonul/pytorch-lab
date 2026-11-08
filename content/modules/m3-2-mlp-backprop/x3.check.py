import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        res = {}
        for w0, b0 in ((0.6, 0.9), (2.0, 2.0)):
            for loss in ("mse", "ce"):
                out = need(ns, "train_neuron")(w0, b0, 1.0, 0.0, loss, 0.15, 300)
                res[(w0, loss)] = out
                rows.append([f"w = b = {w0}" if w0 == 2.0 else "w = 0.6, b = 0.9", loss.upper(), f"{out[0]:.3f}", f"{out[50]:.3f}", f"{out[150]:.3f}", f"{out[-1]:.3f}"])
        ensure(abs(res[(0.6, "mse")][0] - 1 / (1 + np.exp(-1.5))) < 1e-9, "ilk çıktı")
        ensure(res[(2.0, "mse")][150] > 0.8 and res[(2.0, "ce")][150] < 0.2, "doymuş başlangıçta MSE takılır, CE hızla öğrenir")
        R_.table("Nielsen örneği: girdi 1, hedef 0, η = 0.15", ["başlangıç", "kayıp", "adım 0", "adım 50", "adım 150", "adım 300"], rows,
                 note="Başlangıç çıktısı 0.98 (yanlış ve doymuş) iken MSE'nin δ = (a − y)σ'(z) terimi σ' ≈ 0.02 yüzünden çok küçüktür: insan 'çok yanlışken hızlı öğrenir', bu nöron en yanlışken en yavaş öğrenir. Çapraz entropide σ' sadeleşir: δ = a − y, hata büyükse gradyan büyük. Sigmoid/softmax çıkışlarıyla CE kullanmanın gerekçesi budur.")
    R_.test("öğrenme yavaşlaması", t)
