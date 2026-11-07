import numpy as np
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    rng = np.random.default_rng(0)
    T = rng.random((5, 5)); T /= T.sum(0, keepdims=True)
    p0 = np.eye(5)[2]

    def ns_():
        close(need(ns, "n_step")(T, p0, 7), np.linalg.matrix_power(T, 7) @ p0, what="T^7 p0")
        pi = need(ns, "stationary")(T)
        close(T @ pi, pi, atol=1e-12, what="Tπ = π"); close(pi.sum(), 1.0, what="toplam 1")
    R.test("n_step ve stationary", ns_)

    def tv():
        c = need(ns, "tv_curve")(T, p0, 20)
        want = [0.5 * np.abs(np.linalg.matrix_power(T, t) @ p0 - ns["stationary"](T)).sum() for t in range(21)]
        close(c, np.array(want), atol=1e-10, what="TV eğrisi")
        lam2 = sorted(np.abs(np.linalg.eigvals(T)))[-2]
        R.table("Karışma (mixing)", ["t", "TV uzaklığı"], [[t, f"{c[t]:.2e}"] for t in (0, 1, 2, 5, 10, 20)],
                note=f"Uzaklık kabaca |λ₂|^t = {lam2:.3f}^t hızıyla azalır: ikinci en büyük özdeğer karışma hızını belirler.")
    R.test("tv_curve (özdeğer ayrışımıyla)", tv)

    def ar():
        X = need(ns, "ar1_paths")(0.8, 1.0, 200, 4000, 1)
        ensure(X.shape == (4000, 200) and np.all(X[:, 0] == 0), "şekil ve x0 = 0")
        m, v = need(ns, "ensemble_stats")(X)
        theory = (1 - 0.64 ** np.arange(200)) / (1 - 0.64)
        close(v[1:], theory[1:], rtol=0.1, what="Var x_t = (1 − φ^{2t})/(1 − φ²)")
        ensure(np.abs(m).max() < 0.12, "topluluk ortalaması ≈ 0")
        return f"t = 1: var {v[1]:.3f}, t = 199: var {v[199]:.3f} (durağan değer {1 / 0.36:.3f}) — başlangıçta durağan değil, sonra WSS"
    R.test("ar1_paths ve ensemble_stats", ar)
