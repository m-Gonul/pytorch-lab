import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("dropout_np / mc_predict: döngü yok", lambda: (max_loops(src, 0, func="dropout_np"), max_loops(src, 0, func="mc_predict")))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)

    def dn():
        X = rng.normal(size=(50, 40))
        Y, m = need(ns, "dropout_np")(X, 0.4, np.random.default_rng(1))
        want = np.random.default_rng(1).random(X.shape) >= 0.4
        ensure(np.array_equal(np.asarray(m, bool), want), "maske rng.random(X.shape) >= p olmalı")
        close(Y, X * want / 0.6)
    R.test("dropout_np", dn)

    def mc():
        W1, b1 = rng.normal(size=(3, 16)), rng.normal(size=16) * 0.1
        W2, b2 = rng.normal(size=(16, 4)), rng.normal(size=4) * 0.1
        X = rng.normal(size=(10, 3))
        mean, std = need(ns, "mc_predict")(W1, b1, W2, b2, X, 0.5, 200, np.random.default_rng(5))
        r = np.random.default_rng(5)
        H = np.maximum(0, X @ W1 + b1)
        M = r.random((200,) + H.shape) >= 0.5
        P = []
        for t in range(200):
            s = (H * M[t] / 0.5) @ W2 + b2
            e = np.exp(s - s.max(1, keepdims=True)); P.append(e / e.sum(1, keepdims=True))
        P = np.array(P)
        close(mean, P.mean(0)); close(std, P.std(0))
        return f"ortalama tahmin std'si (belirsizlik): {std.mean():.3f}"
    R.test("mc_predict: tek çağrıda T maske, döngülü referansla aynı", mc)

    def es():
        E = need(ns, "EarlyStopper", "sınıf")(2)
        stops = [E.step(v) for v in [0.9, 0.7, 0.6, 0.62, 0.65]]
        ensure(stops == [False, False, False, False, True], f"step sonuçları {stops}")
        ensure(E.best_epoch == 2 and abs(E.best - 0.6) < 1e-12, f"best {E.best}, best_epoch {E.best_epoch}")
        E2 = need(ns, "EarlyStopper", "sınıf")(1, min_delta=0.05)
        ensure([E2.step(v) for v in [1.0, 0.97]] == [False, True], "min_delta")
    R.test("EarlyStopper", es)
