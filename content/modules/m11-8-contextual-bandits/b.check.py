import random
import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def blr():
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (30, 4))
        r = X @ np.array([1.0, -2, 0.5, 0]) + 0.3 * rng.normal(0, 1, 30)
        m = need(ns, "BayesLinReg")(4, 2.0, 0.3)
        for x, y in zip(X, r):
            m.update(list(x), float(y))
        Sig = np.linalg.inv(X.T @ X / 0.09 + np.eye(4) / 4)
        mu = Sig @ X.T @ r / 0.09
        close(np.array(m.Sigma), Sig, atol=1e-9, what="Σ (Sherman–Morrison = kapalı form)")
        q = [0.3, -1.0, 2.0, 0.5]
        close(m.mean(q), float(np.array(q) @ mu), atol=1e-9, what="öngörü ortalaması")
        close(m.var(q), float(np.array(q) @ Sig @ np.array(q)), atol=1e-9, what="parametre belirsizliği x'Σx")
    R_.test("BayesLinReg", blr)

    def sim():
        res = {p: sum(need(ns, "news")(5000, p, s) for s in range(4)) / 4 for p in ("random", "ucb1", "linucb")}
        ensure(res["linucb"] > res["ucb1"] > res["random"], f"sıralama: {res}")
        R_.table("Haber önerisi, 5 makale, 5000 kullanıcı (4 tohum)", ["politika", "tıklama", "CTR"],
                 [[k, f"{v:.0f}", f"{v / 5000:.3f}"] for k, v in res.items()],
                 note="UCB1 tek bir 'ortalama en iyi' makaleye yakınsar; LinUCB her kullanıcıya ilgisine göre makale seçer.")
    R_.test("LinUCB vs UCB1", sim)
