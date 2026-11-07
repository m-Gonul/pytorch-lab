import numpy as np
import pandas as pd
from labcheck import close, no_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch/sklearn kullanılmıyor", lambda: no_imports(src, {"torch", "sklearn"}))
    rng = np.random.default_rng(4)
    C, d = 3, 4
    y = rng.integers(0, C, 150)
    X = rng.normal(size=(150, d)) * (1 + y[:, None]) + y[:, None] * 1.5

    def fit():
        lp, mu, var = need(ns, "fit_gnb")(X, y, C)
        for c in range(C):
            close(mu[c], X[y == c].mean(0), what=f"μ sınıf {c}")
            close(var[c], X[y == c].var(0), rtol=1e-6, atol=1e-8, what=f"σ² sınıf {c}")
        close(np.exp(lp), np.bincount(y) / 150, what="prior")
    R.test("fit_gnb: sınıf ortalaması, MLE varyansı, prior", fit)

    def pred():
        m = need(ns, "fit_gnb")(X, y, C)
        L = need(ns, "predict_log_proba_gnb")(m, X)
        ensure(np.asarray(L).shape == (150, C), f"shape {np.asarray(L).shape}")
        close(np.exp(L).sum(1), np.ones(150), what="olasılık toplamı")
        lp, mu, var = m
        ref = np.stack([lp[c] - 0.5 * (np.log(2 * np.pi * var[c]) + (X - mu[c]) ** 2 / var[c]).sum(1) for c in range(C)], 1)
        ref -= np.log(np.exp(ref - ref.max(1, keepdims=True)).sum(1, keepdims=True)) + ref.max(1, keepdims=True)
        close(L, ref, rtol=1e-7, atol=1e-9)
        return f"eğitim doğruluğu {(np.argmax(L, 1) == y).mean():.3f}"
    R.test("predict_log_proba_gnb normalize ve doğru", pred)

    def tab():
        df = pd.DataFrame({"hava": list("ggyygy"), "oyna": list("EEHEHH")})
        t = need(ns, "cond_table")(df, "hava", "oyna", alpha=1.0)
        ensure(isinstance(t, pd.DataFrame), "DataFrame bekleniyordu")
        close(t.loc[["E", "H"], ["g", "y"]].to_numpy(dtype=float), [[0.6, 0.4], [0.4, 0.6]])
        t0 = need(ns, "cond_table")(df, "hava", "oyna", alpha=0.0)
        close(t0.loc["E", "g"], 2 / 3, what="α = 0")
    R.test("cond_table: Laplace'lı koşullu olasılık tablosu", tab)
