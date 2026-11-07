import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("softmax_rows/softmax_loss: döngü yok", lambda: (max_loops(src, 0, func="softmax_rows"), max_loops(src, 0, func="softmax_loss")))
    R.test("fit_softmax: yalnız adım döngüsü", lambda: max_loops(src, 1, func="fit_softmax"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(2)

    def sm():
        Z = rng.normal(size=(6, 4)) * 5
        Z[0] += 1000
        with np.errstate(over="raise", invalid="raise"):
            P = need(ns, "softmax_rows")(Z)
        E = np.exp(Z - Z.max(1, keepdims=True))
        close(P, E / E.sum(1, keepdims=True))
    R.test("softmax_rows kararlı (+1000 satırı dahil)", sm)

    def loss():
        X, y = rng.normal(size=(20, 5)), rng.integers(0, 3, 20)
        W = rng.normal(size=(5, 3)) * 0.1
        L, dW = need(ns, "softmax_loss")(W, X, y, reg=0.1)
        Z = X @ W
        lse = np.log(np.exp(Z).sum(1))
        close(L, (lse - Z[np.arange(20), y]).mean() + 0.1 * (W ** 2).sum(), what="kayıp")
        h, num = 1e-6, np.zeros_like(W)
        for i in range(5):
            for j in range(3):
                E = np.zeros_like(W); E[i, j] = h
                num[i, j] = (need(ns, "softmax_loss")(W + E, X, y, 0.1)[0] - need(ns, "softmax_loss")(W - E, X, y, 0.1)[0]) / (2 * h)
        close(dW, num, rtol=1e-5, atol=1e-7, what="dW (sayısal gradyan)")
    R.test("softmax_loss: kayıp ve dW (reg dahil) sayısal gradyanla aynı", loss)

    def fit():
        centers = np.array([[0, 3], [-3, -2], [3, -2]])
        y = np.repeat(np.arange(3), 60)
        X = np.column_stack([centers[y] + rng.normal(size=(180, 2)), np.ones(180)])
        W = need(ns, "fit_softmax")(X, y, 3)
        acc = (np.argmax(X @ W, 1) == y).mean()
        ensure(acc >= 0.9, f"doğruluk {acc:.2f}")
        return f"3 sınıf, doğruluk {acc:.3f}"
    R.test("fit_softmax 3 kümeyi ayırıyor", fit)
