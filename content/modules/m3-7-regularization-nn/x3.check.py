import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        g = [np.array([0.9, 100.0])]
        v = need(ns, "clip_value")(g, 1.0)[0]
        n, n0 = need(ns, "clip_global_norm")(g, 1.0)
        close(v, [0.9, 1.0], what="değere göre: yön değişir")
        close(n[0], np.array([0.9, 100.0]) / np.hypot(0.9, 100), what="norma göre: yön korunur")
        close(n0, np.hypot(0.9, 100), what="önceki norm")
        gs = [np.ones((2, 2)), np.ones(4) * 2]
        cl, nn = ns["clip_global_norm"](gs, 1.0)
        close(nn, np.sqrt(4 + 16), what="global norm tüm parametreler üzerinden")
        close(np.sqrt(sum(np.sum(x * x) for x in cl)), 1.0, what="kırpılmış global norm = c")
        rng = np.random.default_rng(0)
        rows = []
        for scale in (0.5, 5, 50):
            G = [rng.normal(size=20) * np.r_[np.full(10, 1.0), np.full(10, scale)]]
            cv = ns["clip_value"](G, 1.0)[0]
            cn = ns["clip_global_norm"](G, 1.0)[0][0]
            cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            rows.append([scale, f"{cos(cv, G[0]):.3f}", f"{cos(cn, G[0]):.3f}"])
        ensure(float(rows[2][1]) < float(rows[0][1]) and float(rows[2][2]) > 0.999, "değere göre kırpma büyük bileşenler baskınken yönü bozar")
        R_.table("Bileşenlerin yarısı s kat büyük bir gradyanda kırpma sonrası yön benzerliği (c = 1)", ["s", "cos(değere göre, g)", "cos(norma göre, g)"], rows,
                 note="Değere göre kırpma büyük bileşenleri 'düzleştirir': gradyan bir işaret vektörüne yaklaşır ve iniş yönünden sapar. Norma göre kırpma yalnız adım boyunu sınırlar. RNN'lerde (M4.3) patlayan gradyanlar genellikle tek bir yönde aniden büyür — norma göre kırpma (Pascanu ve ark.) bu yüzden standarttır; eşik, kırpmasız eğitimdeki tipik normun birkaç katı seçilir.")
    R_.test("gradyan kırpma", t)
