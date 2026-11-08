import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("rrelu", "prelu_grads", "elu", "mish", "dead_fraction")])

    def t():
        rng = np.random.default_rng(0)
        z = np.array([-2.0, -0.5, 0.0, 1.5])
        close(need(ns, "rrelu")(z, False, rng), [-2 * (1 / 8 + 1 / 3) / 2, -0.5 * (1 / 8 + 1 / 3) / 2, 0, 1.5], what="RReLU test")
        tr = ns["rrelu"](np.full(10000, -1.0), True, rng)
        ensure(-1 / 3 <= tr.min() and tr.max() <= -1 / 8 and abs(tr.mean() + (1 / 8 + 1 / 3) / 2) < 0.01, "RReLU eğitim: α ~ U(1/8, 1/3)")
        zz = rng.normal(size=50)
        go = rng.normal(size=50)
        dz, da = need(ns, "prelu_grads")(zz, 0.25, go)
        f = lambda a: np.sum(go * np.where(zz > 0, zz, a * zz))
        close(da, (f(0.25 + 1e-6) - f(0.25)) / 1e-6, atol=1e-4, what="∂/∂α")
        close(dz, go * np.where(zz > 0, 1, 0.25), what="∂/∂z")
        close(need(ns, "mish")(np.array([0.0, 1.0, -1.0])), [0, np.tanh(np.log1p(np.e)), -np.tanh(np.log1p(np.exp(-1)))], what="Mish")
        close(need(ns, "elu")(np.array([-1.0, 2.0])), [np.exp(-1) - 1, 2], what="ELU")
        X = rng.normal(size=(512, 20))
        W = rng.normal(0, np.sqrt(2 / 20), (20, 100))
        b = np.zeros(100)
        rows = []
        for shift in (0.0, -1.0, -2.0, -4.0):
            rows.append([shift, f"{need(ns, 'dead_fraction')(X, W, b + shift * np.sqrt(np.sum(W ** 2, 0))):.2f}"])
        ensure(float(rows[0][1]) == 0.0 and float(rows[3][1]) > 0.9, "ölü nöron oranı bias kaydıkça artar")
        R_.table("ReLU katmanı: bias, ön-aktivasyon std'sinin kaç katı negatife kaydı", ["bias kayması (σ cinsinden)", "ölü nöron oranı (512 örneğin hiçbirinde aktif değil)"], rows,
                 note="Büyük bir gradyan adımı bir ReLU'nun ağırlıklarını tüm eğitim verisi için z < 0 olacak şekilde iterse türevi her yerde 0 olur ve bir daha asla güncellenmez: 'ölü ReLU'. Yüksek öğrenme oranlarında ağın yarısı ölebilir. Sızıntılı varyantlar (α > 0) negatif bölgede de gradyan taşır; Xu ve ark. (2015) bu yüzden hep ReLU'dan iyi buldu. RReLU ayrıca düzenleyici, PReLU büyük veride güçlü.")
    R_.test("ReLU ailesi ve ölü nöronlar", t)
