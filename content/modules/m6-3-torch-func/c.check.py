import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    X, y, w = rng.normal(size=(40, 5)), rng.integers(0, 2, 40), rng.normal(size=5)
    loss_i = lambda w, i: -(y[i] * np.log(1 / (1 + np.exp(-X[i] @ w))) + (1 - y[i]) * np.log(1 - 1 / (1 + np.exp(-X[i] @ w))))
    loss = lambda w: np.mean([loss_i(w, i) for i in range(40)])

    def psg():
        G = need(ns, "per_sample_grads")(w, X, y)
        h = 1e-6
        for i in (0, 7, 39):
            num = [(loss_i(w + h * e, i) - loss_i(w - h * e, i)) / (2 * h) for e in np.eye(5)]
            close(G[i], num, atol=1e-6, what=f"örnek {i}")
    R.test("per_sample_grads = sayısal (örnek başı)", psg)

    def hs():
        H = need(ns, "hessian")(w, X)
        h = 1e-4
        g = lambda w: need(ns, "per_sample_grads")(w, X, y).mean(0)
        num = np.stack([(g(w + h * e) - g(w - h * e)) / (2 * h) for e in np.eye(5)], 1)
        close(H, num, atol=1e-6)
        ensure(np.all(np.linalg.eigvalsh(H) >= -1e-12), "Hessian PSD olmalı (konveks kayıp)")
    R.test("hessian = gradyanın Jacobian'ı, PSD", hs)

    def hv():
        v = rng.normal(size=5)
        close(need(ns, "hvp")(w, X, v), ns["hessian"](w, X) @ v)
    R.test("hvp = H v (H kurulmadan)", hv)
