import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("train: en fazla 2 katman döngü", lambda: max_loops(src, 2, func="train", why="örnekler üzerinde döngü olmamalı"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 1.0 + 0.05 * rng.normal(size=400)

    def full_batch():
        w, b, L = need(ns, "train")(X, y, lr=0.1, batch_size=400, epochs=1)
        e = -y
        close(w, -0.1 * 2 * X.T @ e / 400, what="tam batch tek adım w")
        close(b, -0.1 * 2 * e.mean(), what="tam batch tek adım b")
    R.test("batch_size = n, beta = 0: tek tam GD adımı", full_batch)

    def conv():
        w, b, L = need(ns, "train")(X, y, lr=0.05, batch_size=16, epochs=40, beta=0.0)
        L = np.asarray(L)
        ensure(L.shape == (40,), f"losses shape {L.shape}")
        Xt = np.column_stack([X, np.ones(400)])
        th = np.linalg.lstsq(Xt, y, rcond=None)[0]
        close(w, th[:3], atol=0.03, rtol=0, what="w"); close(b, th[3], atol=0.03, rtol=0, what="b")
        return f"son epoch kaybı {L[-1]:.5f}"
    R.test("mini-batch SGD en küçük kareler çözümüne yaklaşıyor", conv)

    def det():
        a = need(ns, "train")(X, y, lr=0.05, batch_size=32, epochs=3, seed=5)
        b_ = need(ns, "train")(X, y, lr=0.05, batch_size=32, epochs=3, seed=5)
        close(a[0], b_[0], what="aynı seed")
    R.test("seed ile tekrarlanabilir", det)

    def ill():
        Xi = rng.normal(size=(400, 2)) * np.array([1.0, 10.0])
        yi = Xi @ np.array([3.0, 0.2])
        _, _, L0 = need(ns, "train")(Xi, yi, lr=0.004, batch_size=400, epochs=60, beta=0.0)
        _, _, L9 = need(ns, "train")(Xi, yi, lr=0.004, batch_size=400, epochs=60, beta=0.9)
        ensure(L9[-1] < 0.1 * L0[-1], f"β = 0.9: {L9[-1]:.4g}, β = 0: {L0[-1]:.4g} — momentum belirgin daha iyi olmalı")
        return f"60 adım sonra kayıp: β = 0 → {L0[-1]:.4g}, β = 0.9 → {L9[-1]:.4g}"
    R.test("kötü koşullu problemde momentum hızlandırıyor", ill)
