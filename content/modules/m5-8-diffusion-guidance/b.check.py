import math
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t():
        close(need(ns, "cfg")([0.2, -1.0], [0.5, 1.0], 3), [1.1, 5.0], what="CFG")
        close(ns["cfg"]([0.2], [0.5], 0), [0.2], what="s = 0 koşulsuz")
        close(ns["cfg"]([0.2], [0.5], 1), [0.5], what="s = 1 koşullu")
        close(need(ns, "classifier_guided")(0.4, 0.75, 2, 0.3), 0.1, what="sınıflandırıcı yönlendirmesi")
        close(need(ns, "guided_gauss")(0, 1, 2, 1, 3), (1.5, 0.25), what="yönlendirilmiş Gauss")
        close(ns["guided_gauss"](0, 1, 2, 1, 1), (1.0, 0.5), what="s = 1: gerçek posterior")
        close(need(ns, "cosine_abar")(0, 1000), 1.0, what="t = 0")
        close(ns["cosine_abar"](500, 1000), math.cos(0.508 / 1.008 * math.pi / 2) ** 2 / math.cos(0.008 / 1.008 * math.pi / 2) ** 2, what="kosinüs ortası")
        close(need(ns, "linear_abar")(2, 1000), (1 - 1e-4) * (1 - 1e-4 - (0.02 - 1e-4) / 999), what="doğrusal")
        close(need(ns, "snr")(0.75), 3.0, what="SNR")
        rows = []
        for s in (0, 0.5, 1, 2, 4, 10):
            m, v = ns["guided_gauss"](0, 1, 2, 1, s)
            rows.append([s, f"{m:.3f}", f"{math.sqrt(v):.3f}"])
        R_.table("Önsel N(0, 1), olabilirlik N(x; 2, 1): p(x)·p(y|x)^s", ["s", "ortalama", "std"], rows,
                 note="Yönlendirme ölçeği olabilirliği 'keskinleştirir': s = 1 gerçek koşullu dağılım, s > 1 dağılımı olabilirliğin tepesine doğru iter ve daraltır (kesinlik–çeşitlilik ödünleşimi), s < 1 önsele doğru yumuşatır.")
    R_.test("formüller", t)
