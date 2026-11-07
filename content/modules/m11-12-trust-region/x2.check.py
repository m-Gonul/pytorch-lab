import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        tg = np.linspace(-2, 4, 601)
        th0 = np.array([0.2, -0.1, 0.0])
        d = np.array([1.0, -0.6, -0.2])
        A = np.array([1.0, -0.8, 0.1])
        cpi, clip, klp, kl = need(ns, "surrogates")(tg, th0, d, A)
        ensure(np.all(clip <= cpi + 1e-12), "L^CLIP ≤ L^CPI (kötümser)")
        i0 = np.argmin(np.abs(tg))
        close(clip[i0], cpi[i0], what="t = 0'da eşit")
        h = tg[1] - tg[0]
        close((clip[i0 + 1] - clip[i0 - 1]) / (2 * h), (cpi[i0 + 1] - cpi[i0 - 1]) / (2 * h), rtol=1e-3, what="t = 0'da aynı gradyan")
        tc, tk, tp = tg[clip.argmax()], tg[klp.argmax()], tg[cpi.argmax()]
        ensure(tc < tp, "L^CLIP'in en iyisi L^CPI'ninkinden önce")
        R_.table("Yön d boyunca vekil amaçlar (ε = 0.2, β = 1)", ["amaç", "en iyi t", "orada KL", "orada L^CPI"],
                 [["L^CPI (kısıtsız)", f"{tp:.2f}", f"{kl[cpi.argmax()]:.3f}", f"{cpi.max():.4f}"], ["L^CLIP", f"{tc:.2f}", f"{kl[clip.argmax()]:.3f}", f"{cpi[clip.argmax()]:.4f}"],
                  ["L^CPI − β·KL", f"{tk:.2f}", f"{kl[klp.argmax()]:.3f}", f"{cpi[klp.argmax()]:.4f}"]],
                 note="L^CPI tek başına bu yönde sınırsız ilerlemeyi ödüllendirir (aralığın ucunda en büyük); kırpma ve KL cezası adımı politikaya yakın bir bölgede tutar (PPO makalesi Şekil 2).")
    R_.test("vekil amaçlar", t)
