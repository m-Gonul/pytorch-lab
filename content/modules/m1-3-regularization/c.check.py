import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 6)) * np.array([1, 2, 0.5, 1, 3, 1])
    y = X @ rng.normal(size=6) + 2.0 + 0.1 * rng.normal(size=60)
    Xc, yc = X - X.mean(0), y - y.mean()

    def rd():
        for lam in (0.0, 0.5, 20.0):
            w, b = need(ns, "ridge")(X, y, lam)
            want = np.linalg.solve(Xc.T @ Xc + lam * np.eye(6), Xc.T @ yc)
            close(w, want, what=f"λ = {lam} w"); close(b, y.mean() - X.mean(0) @ want, what=f"λ = {lam} b")
        w0, b0 = need(ns, "ridge")(np.array([[1., 0], [0, 1], [1, 1]]), np.array([1., 2, 3]), 0.0)
    R.test("ridge: bias'lı kapalı form (λ = 0, 0.5, 20)", rd)

    def rp():
        lams = np.array([0.0, 0.1, 1.0, 10.0, 1e4])
        P = need(ns, "ridge_path")(X, y, lams)
        ensure(np.asarray(P).shape == (5, 6), f"shape {np.asarray(P).shape}")
        for i, lam in enumerate(lams):
            close(P[i], np.linalg.solve(Xc.T @ Xc + lam * np.eye(6), Xc.T @ yc), rtol=1e-7, atol=1e-9, what=f"λ = {lam}")
        n = np.linalg.norm(P, axis=1)
        ensure(np.all(np.diff(n) < 0), "‖w‖ λ arttıkça azalmalı")
        return "‖w(λ)‖: " + ", ".join(f"{v:.3f}" for v in n)
    R.test("ridge_path tek SVD'den, her λ'da kapalı formla aynı", rp)
