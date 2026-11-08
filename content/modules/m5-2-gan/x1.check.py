import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: max_loops(src, 0, func="minibatch_features"))

    def t():
        rng = np.random.default_rng(0)
        F = rng.normal(size=(4, 3))
        T = rng.normal(size=(3, 2, 5))
        o = need(ns, "minibatch_features")(F, T)
        M = np.einsum("na,abc->nbc", F, T)
        ref = np.array([[sum(np.exp(-np.abs(M[i, b] - M[j, b]).sum()) for j in range(4)) for b in range(2)] for i in range(4)])
        close(o, ref, what="tanım")
        T2 = rng.normal(size=(8, 16, 4))
        div = rng.normal(size=(64, 8))
        col = np.repeat(rng.normal(size=(1, 8)), 64, 0) + 0.01 * rng.normal(size=(64, 8))
        od, oc = ns["minibatch_features"](div, T2), ns["minibatch_features"](col, T2)
        ensure(oc.mean() > 20 * od.mean(), "çökmüş batch'te benzerlik öznitelikleri devasa")
        R_.note(f"Çeşitli batch'te o(x) ≈ {od.mean():.2f} (yalnız kendisi ≈ 1 katkı yapar), neredeyse aynı örneklerden oluşan batch'te ≈ {oc.mean():.1f} (≈ batch boyutu 64). Ayırıcı bu özniteliği gerçek batch'lerde düşük gördüğü için çökmüş batch'i tek bakışta reddeder; üretici çeşitlilik üretmeye zorlanır. ProGAN'ın mini-batch std katmanı (M5.7 x3) aynı fikrin parametresiz sürümüdür.")
    R_.test("mini-batch ayrımı", t)
