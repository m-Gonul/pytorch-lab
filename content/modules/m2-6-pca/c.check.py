import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def sign_fix(V):
    idx = np.argmax(np.abs(V), axis=1)
    return V * np.sign(V[np.arange(len(V)), idx])[:, None]


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 4)) @ rng.normal(size=(4, 4)) + np.array([10.0, -3, 0, 7])
    w, V = np.linalg.eigh(np.cov(X.T))

    def comp():
        P, var, Z = need(ns, "pca")(X, 3)
        ensure(np.asarray(P).shape == (3, 4) and np.asarray(Z).shape == (150, 3), "shape'ler")
        close(P @ P.T, np.eye(3), atol=1e-9, what="ortonormallik")
        close(var, w[::-1][:3], rtol=1e-8, what="varyanslar = özdeğerler")
        close(P, sign_fix(V[:, ::-1].T)[:3], atol=1e-8, what="bileşenler (işaret kuralıyla)")
        close(Z.var(0, ddof=1), w[::-1][:3], rtol=1e-8, what="Z sütun varyansları")
    R.test("bileşenler, varyanslar, işaret kuralı", comp)

    def rec():
        P, var, Z = need(ns, "pca")(X, 4)
        close(need(ns, "reconstruct")(Z, P, X.mean(0)), X, atol=1e-9, what="k = d tam kurma")
        P2, _, Z2 = need(ns, "pca")(X, 2)
        err = ((need(ns, "reconstruct")(Z2, P2, X.mean(0)) - X) ** 2).sum() / (len(X) - 1)
        close(err, w[::-1][2:].sum(), rtol=1e-8, what="k = 2 hatası = atılan özdeğerlerin toplamı")
        return f"k = 2 yeniden kurma hatası {err:.4f} = λ₃ + λ₄"
    R.test("yeniden kurma: kayıp = atılan varyans", rec)
