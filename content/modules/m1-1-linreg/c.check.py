import numpy as np
import pandas as pd
from labcheck import close, max_loops, no_imports, need, ensure, uses_any


def check(ns, R):
    src = ns["__source__"]
    R.test("normal_eq içinde döngü yok", lambda: max_loops(src, 0, func="normal_eq"))
    R.test("fit_gd içinde yalnız adım döngüsü", lambda: max_loops(src, 1, func="fit_gd", why="örnekler üzerinde döngü olmamalı"))
    R.test("torch kullanılmıyor, inv yok", lambda: (no_imports(src, {"torch"}), ensure(not uses_any(src, ["inv"]), "np.linalg.inv yerine lstsq/solve kullan")))
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 4))
    y = X @ np.array([1.5, -2.0, 0.0, 0.7]) - 1.0 + 0.05 * rng.normal(size=80)
    Xt = np.column_stack([X, np.ones(80)])
    th = np.linalg.solve(Xt.T @ Xt, Xt.T @ y)

    def ne():
        w, b = need(ns, "normal_eq")(X, y)
        close(w, th[:4], what="w"); close(b, th[4], what="b")
        w1, b1 = need(ns, "normal_eq")(np.array([[1.], [2.], [3.]]), np.array([2., 4., 5.]))
        close([w1[0], b1], [1.5, 2 / 3], what="elle örnek")
    R.test("normal_eq: 4 özellik ve elle örnek", ne)

    def gd():
        w, b, losses = need(ns, "fit_gd")(X, y, lr=0.1, steps=500)
        losses = np.asarray(losses)
        ensure(losses.shape == (500,), f"losses shape {losses.shape}, (500,) bekleniyordu")
        ensure(np.all(np.diff(losses) <= 1e-12), "kayıp bir adımda arttı (lr = 0.1 ile olmamalı)")
        close(w, th[:4], atol=1e-4, rtol=0, what="GD w"); close(b, th[4], atol=1e-4, rtol=0, what="GD b")
        return f"kayıp {losses[0]:.3f} → {losses[-1]:.5f}"
    R.test("fit_gd normal denkleme yakınsıyor, kayıp monoton", gd)

    def df_():
        df = pd.DataFrame(X, columns=["a", "b", "c", "d"]).assign(hedef=y)
        w, b = need(ns, "fit_df")(df, "hedef")
        ensure(isinstance(w, pd.Series), "w bir pd.Series olmalı")
        ensure(list(w.index) == ["a", "b", "c", "d"], f"indeks {list(w.index)}, özellik isimleri bekleniyordu")
        close(w.to_numpy(), th[:4]); close(b, th[4])
        df2 = df[["hedef", "d", "a", "c", "b"]]
        w2, _ = need(ns, "fit_df")(df2, "hedef")
        close(w2[["a", "b", "c", "d"]].to_numpy(), th[:4], what="sütun sırası değişince")
        return "sütun sırası değişse de isimler doğru katsayıyı taşıyor"
    R.test("fit_df: isimli Series, sütun sırasından bağımsız", df_)
