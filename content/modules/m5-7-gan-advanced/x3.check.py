import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        old = rng.normal(size=(3, 4, 4))
        new = rng.normal(size=(3, 8, 8))
        f = need(ns, "fade_in")(old, new, 0.25)
        close(f[:, 2, 3], 0.75 * old[:, 1, 1] + 0.25 * new[:, 2, 3], what="yumuşak geçiş")
        close(ns["fade_in"](old, new, 1.0), new, what="α = 1 → yalnız yeni katman")
        X = rng.normal(size=(8, 4, 5, 5))
        Y = need(ns, "minibatch_std")(X)
        ensure(Y.shape == (8, 5, 5, 5) and np.allclose(Y[:, 4], Y[0, 4, 0, 0]), "sabit ek kanal")
        close(Y[0, 4, 0, 0], X.std(0).mean(), atol=1e-6, what="batch std'lerinin ortalaması")
        Yc = ns["minibatch_std"](np.repeat(X[:1], 8, 0))
        ensure(Yc[0, 4, 0, 0] < 1e-3, "çeşitliliği olmayan batch → ~0: ayırıcı mod çöküşünü görebilir")
        P = need(ns, "pixel_norm")(X * 10)
        close(np.sqrt((P ** 2).mean(1)), np.ones((8, 5, 5)), atol=1e-6, what="piksel başına birim RMS")
        rows = []
        for fan in (16, 256, 4096):
            a, b = need(ns, "equalized_step")(fan, 1e-3, 100, 0)
            rows.append([fan, f"{a:.3f}", f"{b:.3f}"])
        ensure(float(rows[2][1]) > 5 * float(rows[0][1]) and abs(float(rows[2][2]) - float(rows[0][2])) < 0.01, "eşitlenmiş lr: göreli adım fan-in'den bağımsız")
        R_.table("Adam benzeri (işaret) güncellemelerle 100 adım sonra etkin ağırlığın göreli değişimi", ["fan-in", "He başlatma", "N(0, 1) + çalışma anı çarpanı c"], rows,
                 note="Adam gradyanı kendi ölçeğine böler: adım boyu parametrenin büyüklüğünden bağımsızdır. He ile başlatılmış küçük ağırlıklar (büyük fan-in) bu yüzden göreli olarak çok daha hızlı değişir; katmanlar farklı hızlarda öğrenir. Eşitlenmiş öğrenme oranı (ProGAN) tüm ağırlıkları N(0, 1) tutar, ölçeği ileri geçişte uygular: her katmanın göreli öğrenme hızı aynı olur.")
    R_.test("ProGAN hileleri", t)
