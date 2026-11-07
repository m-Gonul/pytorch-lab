import numpy as np
import pandas as pd
from labcheck import close, no_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok (comprehension dahil)", lambda: no_loops(src, allow_comprehension=False))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)

    def g():
        X = rng.normal(loc=[1, -2, 5], scale=[1, 3, 0.5], size=(200, 3))
        mu, var = need(ns, "gaussian_mle")(X)
        close(mu, X.mean(0)); close(var, X.var(0))
        close(need(ns, "gaussian_mle")(np.array([[2.0], [4.0], [6.0]]))[1], [8 / 3], what="elle örnek varyans")
    R.test("gaussian_mle: sütun başına ortalama ve MLE varyansı", g)

    def gs():
        grid = np.linspace(0.01, 0.99, 99)
        close(need(ns, "grid_search_p")(np.array([1, 0, 1, 1]), grid), 0.75, atol=1e-9)
        ys = (rng.random(300) < 0.31).astype(int)
        p = need(ns, "grid_search_p")(ys, grid)
        ensure(abs(p - ys.mean()) <= 0.0051, f"ızgara sonucu {p}, veri oranı {ys.mean():.3f}")
        return f"300 atış: ızgara {p:.2f}, kapalı form {ys.mean():.3f}"
    R.test("grid_search_p kapalı forma (oran) en yakın ızgara noktası", gs)

    def cs():
        df = pd.DataFrame({"sinif": ["a", "a", "b", "b", "b"], "boy": [1.0, 3.0, 2.0, 4.0, 6.0]})
        out = need(ns, "class_stats")(df, "sinif", "boy")
        ensure(isinstance(out, pd.DataFrame), "DataFrame bekleniyordu")
        for c in ("count", "mean", "var"):
            ensure(c in out.columns, f"`{c}` sütunu yok")
        close(out.loc[["a", "b"], ["count", "mean", "var"]].to_numpy(dtype=float), [[2, 2, 1], [3, 4, 8 / 3]])
    R.test("class_stats: count, mean, MLE var", cs)
