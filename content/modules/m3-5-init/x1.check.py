import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        close(need(ns, "selu")(np.array([1.0, -1.0])), [1.0507009873554805, 1.0507009873554805 * 1.6732632423543772 * (np.exp(-1) - 1)], what="SELU")
        m, s2 = need(ns, "gauss_moments")(ns["selu"])
        close((m, s2), (0.0, 1.0), atol=1e-4, what="sabit nokta: E[selu(z)] = 0, E[selu(z)²] = 1")
        mr, s2r = ns["gauss_moments"](lambda z: np.maximum(z, 0))
        close((mr, s2r), (1 / np.sqrt(2 * np.pi), 0.5), atol=1e-5, what="ReLU: ortalama 1/√(2π), ikinci moment 1/2")
        rows = []
        cfg = (("SELU + LeCun", ns["selu"], "lecun"), ("SELU + He", ns["selu"], "he"), ("ELU + LeCun", lambda z: np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1), "lecun"),
               ("ReLU + He", lambda z: np.maximum(z, 0), "he"), ("tanh + Glorot", np.tanh, "glorot"))
        res = {}
        for name, f, init in cfg:
            st = need(ns, "deep_stats")(f, init, 50, 256, 0)
            res[name] = st
            rows.append([name] + [f"{st[k][0]:+.2f} / {st[k][1]:.2f}" for k in (0, 9, 49)])
        sl = res["SELU + LeCun"]
        ensure(abs(sl[49][0]) < 0.1 and abs(sl[49][1] - 1) < 0.1, "SELU + LeCun 50 katmanda (0, 1) civarı")
        ensure(res["SELU + He"][49][1] > 1.3, "SELU + He: varyans büyür")
        ensure(res["tanh + Glorot"][49][1] < 0.5, "tanh: aktivasyonlar söner")
        R_.table("256 genişlikli, 50 katmanlı ağ: ortalama / std (katman 1, 10, 50)", ["aktivasyon + başlatma", "katman 1", "katman 10", "katman 50"], rows,
                 note="SELU'nun λ, α sabitleri E[selu(z)] = 0 ve E[selu²] = 1'i sağlayacak şekilde seçilmiştir; LeCun başlatmayla her katmanın ön-aktivasyonu yine yaklaşık N(0, 1) olur — dağılım katmanlar boyunca kendini düzeltir (öz-normalizasyon). He başlatma (2 kat varyans) bu sabit noktayı bozar; ReLU + He varyansı korur ama ortalama pozitif kayar; tanh + Glorot derinlikte söner.")
    R_.test("SELU öz-normalizasyonu", t)
