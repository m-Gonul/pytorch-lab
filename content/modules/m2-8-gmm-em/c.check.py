import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def ref_logpdf(x, mu, S):
    d = len(x); diff = x - mu
    return -0.5 * (d * np.log(2 * np.pi) + np.log(np.linalg.det(S)) + diff @ np.linalg.solve(S, diff))


def check(ns, R):
    src = ns["__source__"]
    for fn in ("log_gauss", "e_step_np", "m_step_np"):
        R.test(f"{fn}: döngü yok", lambda fn=fn: max_loops(src, 0, func=fn))
    R.test("em_np: yalnız iterasyon döngüsü", lambda: max_loops(src, 1, func="em_np"))
    R.test("torch/sklearn/scipy kullanılmıyor", lambda: no_imports(src, {"torch", "sklearn", "scipy"}))
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.multivariate_normal([0, 0, 0], [[1, .5, 0], [.5, 1, .2], [0, .2, 1]], 120), rng.normal([4, 1, -2], 0.6, (80, 3))])
    mus = X[[0, 150]]
    covs = np.stack([np.eye(3) * 2, np.array([[1, .3, 0], [.3, 1, 0], [0, 0, .5]])])

    def lg():
        L = need(ns, "log_gauss")(X, mus, covs)
        ensure(np.asarray(L).shape == (200, 2), f"shape {np.asarray(L).shape}")
        close(L[[0, 7, 199]], [[ref_logpdf(X[i], mus[k], covs[k]) for k in range(2)] for i in (0, 7, 199)])
    R.test("log_gauss tek tek yoğunlukla aynı", lg)

    def ms():
        Rr, _ = need(ns, "e_step_np")(X, np.array([0.3, 0.7]), mus, covs)
        close(Rr.sum(1), np.ones(200), what="responsibility satır toplamı")
        p, m, C = need(ns, "m_step_np")(X, Rr, reg=0.0)
        for k in range(2):
            w = Rr[:, k]
            mk = (w[:, None] * X).sum(0) / w.sum()
            Ck = ((X - mk).T * w) @ (X - mk) / w.sum()
            close(m[k], mk, what=f"μ_{k}"); close(C[k], Ck, what=f"Σ_{k}")
    R.test("e_step_np + m_step_np: ağırlıklı MLE", ms)

    def em():
        p, m, C, lls = need(ns, "em_np")(X, np.array([0.5, 0.5]), mus, covs, 25)
        lls = np.asarray(lls)
        ensure(np.all(np.diff(lls) >= -1e-8), "log-likelihood düştü")
        return f"ll: {lls[0]:.1f} → {lls[-1]:.1f}"
    R.test("em_np: log-likelihood monoton", em)
