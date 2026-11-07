import numpy as np
from labcheck import close, no_imports, need, ensure


def sign_fix(V):
    idx = np.argmax(np.abs(V), axis=1)
    return V * np.sign(V[np.arange(len(V)), idx])[:, None]


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3)) @ np.array([[2.0, 0.5, 0], [0, 1.0, 0.3], [0, 0, 0.4]])

    def cov():
        close(need(ns, "covariance")(X.tolist()), np.cov(X.T))
    R.test("covariance = np.cov", cov)

    def pw():
        lam, v = need(ns, "power_iteration")([[2.0, 1.0], [1.0, 2.0]])
        close([lam, v[0], v[1]], [3, 0.5 ** 0.5, 0.5 ** 0.5])
        lam2, v2 = need(ns, "power_iteration")([[1.0, 0], [0, -4.0]])
        close([lam2, v2[0], v2[1]], [-4, 0, 1], atol=1e-6, what="negatif özdeğer: |λ| en büyük, işaret kuralı")
    R.test("power_iteration: elle örnek ve işaret kuralı", pw)

    def tk():
        C = np.cov(X.T)
        lams, vecs = need(ns, "top_k")(C.tolist(), 3, 500)
        w, V = np.linalg.eigh(C)
        close(lams, w[::-1], rtol=1e-6, what="özdeğerler")
        close(np.array(vecs), sign_fix(V[:, ::-1].T), atol=1e-5, rtol=0, what="özvektörler")
    R.test("top_k (deflation) = eigh", tk)
