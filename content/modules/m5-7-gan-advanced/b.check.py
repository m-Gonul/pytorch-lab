import math
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t():
        close(need(ns, "frechet_1d")(0, 1, 1, 2), 2.0, what="FID 1B")
        close(ns["frechet_1d"](3, 2, 3, 2), 0.0, what="aynı dağılım")
        close(need(ns, "inception_score")([[0.9, 0.1], [0.1, 0.9]]), math.exp(0.9 * math.log(1.8) + 0.1 * math.log(0.2)), what="IS")
        close(ns["inception_score"]([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 3.0, what="mükemmel ve dengeli: sınıf sayısı")
        close(ns["inception_score"]([[1, 0, 0]] * 5), 1.0, what="tek sınıf: 1")
        y = need(ns, "adain")([1, 2, 3, 4], 2, 1)
        close(y[3], 2 * 1.5 / math.sqrt(1.25) + 1, what="AdaIN")
        m = sum(y) / 4
        close((m, math.sqrt(sum((v - m) ** 2 for v in y) / 4)), (1.0, 2.0), what="çıktı ortalama y_b, std y_s")
        close(need(ns, "truncate")([2.0, -1.0], [0.0, 1.0], 0.7), [1.4, -0.4], what="kırpma")
        close(need(ns, "fade_in")([2.0], [4.0], 0.3), [2.6], what="yumuşak geçiş")
        close(need(ns, "he_scale")(512), math.sqrt(2 / 512), what="He ölçeği")
        rows = []
        for mu, s in ((0, 1), (0.5, 1), (0, 0.5), (0, 2), (2, 1)):
            rows.append([mu, s, f"{ns['frechet_1d'](0, 1, mu, s):.2f}"])
        R_.table("Gerçek N(0, 1) ile üretilen N(μ, σ²) arasındaki Fréchet uzaklığı", ["μ", "σ", "FID (1B)"], rows,
                 note="FID ortalama kaymasını ve yayılım farkını birlikte cezalandırır: σ = 0.5 (çeşitlilik kaybı, mod çöküşünün 1B karşılığı) 0.25, σ = 2 (aşırı yayılma) 1.0 ceza alır.")
    R_.test("formüller", t)
