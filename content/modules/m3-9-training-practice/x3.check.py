import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        X = np.arange(64.0).reshape(1, 64)
        A = need(ns, "shift_augment")(X)
        ensure(A.shape == (5, 64) and (A[0] == X[0]).all(), "orijinal + 4 kaydırma")
        I = A.reshape(5, 8, 8)
        ensure(I[1, 1:, :].tolist() == X.reshape(8, 8)[:-1, :].tolist() and (I[1, 0] == 0).all(), "aşağı kaydırma, sıfır dolgu")
        res = [(need(ns, "few_shot_accuracy")(False, s), ns["few_shot_accuracy"](True, s)) for s in range(5)]
        a = np.mean([r[0] for r in res])
        b = np.mean([r[1] for r in res])
        ensure(b > a + 0.02, "artırma yardımcı olmalı")
        R_.table("digits, sınıf başına 3 eğitim örneği, 1-NN (5 tohum)", ["veri", "test doğruluğu"], [["30 örnek", f"{a:.3f}"], ["+ ±1 piksel kaydırmalar (150)", f"{b:.3f}"]],
                 note="Kaydırılmış kopyalar, test rakamı biraz kaymış yazıldığında da yakın bir komşu bulunmasını sağlar: değişmezlik bilgisini veriye gömmek. (Lojistik regresyonda aynı artırma 8×8'lik küçük görüntülerde sınıfları bulandırıp zarar verebilir — artırma modelle birlikte düşünülmeli.) CNN'ler kaydırma eşdeğerliğini mimariyle (M4.1) verir.")
    R_.test("veri artırma", t)
