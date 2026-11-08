import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("group_norm", "conv3x3", "resblock", "attn2d")])

    def t():
        rng = np.random.default_rng(0)
        C, H = 8, 6
        x = rng.normal(2, 3, (C, H, H))
        g = need(ns, "group_norm")(x, 4)
        gg = g.reshape(4, -1)
        close((gg.mean(1), gg.std(1)), (np.zeros(4), np.ones(4)), atol=1e-4, what="GroupNorm grupları")
        close(ns["group_norm"](x * 5 + 1, 4), g, atol=1e-4, what="ölçek ve kaymadan bağımsız")
        W, b = rng.normal(size=(5, C, 3, 3)), rng.normal(size=5)
        y = need(ns, "conv3x3")(x, W, b)
        close(y[2, 3, 3], np.sum(W[2] * x[:, 2:5, 2:5]) + b[2], what="konvolüsyon")
        P = {"G": 4, "W1": rng.normal(size=(C, C, 3, 3)) * 0.1, "b1": np.zeros(C), "Wt": rng.normal(size=(C, 16)), "bt": np.zeros(C), "W2": np.zeros((C, C, 3, 3)), "b2": np.zeros(C)}
        te = rng.normal(size=16)
        close(need(ns, "resblock")(x, te, P), x, what="sıfır başlatılmış ikinci konv. → blok özdeşlik")
        P["W2"] = rng.normal(size=(C, C, 3, 3)) * 0.1
        r1, r2 = ns["resblock"](x, te, P), ns["resblock"](x, te + 1.0, P)
        ensure(np.abs(r1 - r2).max() > 1e-3, "zaman embedding'i çıktıyı değiştirir")
        A = {"G": 4, "Wq": rng.normal(size=(C, C)), "Wk": rng.normal(size=(C, C)), "Wv": rng.normal(size=(C, C)), "Wo": np.zeros((C, C))}
        close(need(ns, "attn2d")(x, A), x, what="sıfır W_o → özdeşlik")
        A["Wo"] = rng.normal(size=(C, C)) * 0.1
        xp = x.reshape(C, -1)[:, rng.permutation(H * H)].reshape(C, H, H)
        ensure(ns["attn2d"](x, A).shape == x.shape, "şekil")
        R_.note("Difüzyon U-Net'lerinde her artık bloğun ve attention bloğunun çıkış izdüşümü sıfırla başlatılır: ağ eğitime özdeşlik + giriş/çıkış konvolüsyonlarından oluşan sığ bir ağ olarak başlar. GroupNorm batch'ten bağımsızdır (küçük batch ve EMA ile güvenli) ve girdi ölçeğini unutur — zaman embedding'i bu yüzden normalizasyondan sonra eklenir ki 'ne kadar gürültü var' bilgisi silinmesin.")
    R_.test("U-Net blokları", t)
