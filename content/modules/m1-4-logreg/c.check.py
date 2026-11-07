import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("sigmoid_np/predict içinde döngü yok", lambda: (max_loops(src, 0, func="sigmoid_np"), max_loops(src, 0, func="predict")))
    R.test("fit_logreg içinde yalnız adım döngüsü", lambda: max_loops(src, 1, func="fit_logreg"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))

    def sg():
        z = np.array([-1000.0, -50, -1, 0, 2, 50, 1000])
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            s = need(ns, "sigmoid_np")(z)
        want = np.array([0.0, 1.9287498479639178e-22, 0.2689414213699951, 0.5, 0.8807970779778823, 1.0, 1.0])
        close(s, want, rtol=1e-9, atol=1e-30)
        return "±1000'de bile uyarı/taşma yok"
    R.test("sigmoid_np kararlı (taşma/geçersiz değer hatası yok)", sg)

    rng = np.random.default_rng(5)
    X = np.concatenate([rng.normal(-1.2, 1, (120, 3)), rng.normal(1.2, 1, (120, 3))])
    y = np.repeat([0, 1], 120)

    def one():
        w, b, L = need(ns, "fit_logreg")(np.array([[-1.0], [0.0], [2.0]]), np.array([0, 1, 1]), lr=1.0, steps=1)
        close(np.asarray(L)[0], 0.6931471805599453, what="ilk kayıp (w = b = 0)")
        close([float(np.ravel(w)[0]), b], [-1.0 * (0.5 * -1 + -0.5 * 0 + -0.5 * 2) / 3, -1.0 * (0.5 - 0.5 - 0.5) / 3], what="tek adım sonrası (w, b)")
    R.test("tek adım: sıfırdan başlayıp doğru gradyanla güncelliyor", one)

    def tr():
        w, b, L = need(ns, "fit_logreg")(X, y, lr=0.5, steps=500)
        L = np.asarray(L)
        ensure(L.shape == (500,), f"losses shape {L.shape}")
        ensure(L[-1] < L[0], "kayıp azalmadı")
        acc = (np.asarray(need(ns, "predict")(X, w, b)) == y).mean()
        ensure(acc >= 0.9, f"doğruluk {acc:.2f}")
        return f"doğruluk {acc:.3f}, kayıp {L[0]:.3f} → {L[-1]:.3f}"
    R.test("eğitim + predict", tr)
