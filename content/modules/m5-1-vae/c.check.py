import numpy as np
from labcheck import close, no_loops, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("ae_grads döngüsüz", lambda: max_loops(src, 0, func="ae_grads"))
    R.test("pca_error döngüsüz", lambda: max_loops(src, 0, func="pca_error"))
    R.test("train_linear_ae: tek döngü", lambda: max_loops(src, 1, func="train_linear_ae"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 4)); We = rng.normal(size=(4, 2)); Wd = rng.normal(size=(2, 4))

    def gr():
        L, dWe, dWd = need(ns, "ae_grads")(X, We, Wd)
        f = lambda We_, Wd_: ((X @ We_ @ Wd_ - X) ** 2).sum() / len(X)
        close(L, f(We, Wd))
        h = 1e-6
        for W, dW, which in ((We, dWe, 0), (Wd, dWd, 1)):
            num = np.zeros_like(W)
            for idx in np.ndindex(W.shape):
                Wp = W.copy(); Wp[idx] += h; Wm = W.copy(); Wm[idx] -= h
                num[idx] = (f(Wp, Wd) - f(Wm, Wd)) / (2 * h) if which == 0 else (f(We, Wp) - f(We, Wm)) / (2 * h)
            close(dW, num, atol=1e-4, what="dWe" if which == 0 else "dWd")
    R.test("ae_grads = sayısal türev", gr)

    def pe():
        Y = X - X.mean(0)
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        best = Y - (U[:, :2] * s[:2]) @ Vt[:2]
        close(need(ns, "pca_error")(Y, 2), (best ** 2).sum() / len(Y))
    R.test("pca_error = rank-2 SVD hatası", pe)

    def tr():
        X = rng.normal(size=(200, 5)) * np.array([3, 2, 1, 0.3, 0.1])
        X = X @ np.linalg.qr(rng.normal(size=(5, 5)))[0]; X -= X.mean(0)
        We, Wd, L = need(ns, "train_linear_ae")(X, 2)
        opt = need(ns, "pca_error")(X, 2)
        ensure(L < opt * 1.01, f"AE hatası {L:.4f}, PCA optimumu {opt:.4f}")
        P = Wd.T @ np.linalg.pinv(Wd.T)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Q = Vt[:2].T @ Vt[:2]
        ensure(np.abs(P - Q).max() < 0.05, "çözücünün satır uzayı ilk 2 PCA yönüyle örtüşmeli")
        return f"AE {L:.4f} ≈ PCA {opt:.4f}; aynı alt uzay"
    R.test("doğrusal AE = PCA", tr)
