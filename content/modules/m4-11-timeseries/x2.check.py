import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("ensemble_stats", "wss_gap", "time_mean")])

    def t():
        rng = np.random.default_rng(0)
        R, T = 4000, 60
        t = np.arange(T)
        w = 0.3
        procs = {
            "rastgele fazlı kosinüs": np.cos(w * t + rng.uniform(0, 2 * np.pi, (R, 1))),
            "rastgele genlikli kosinüs": rng.normal(size=(R, 1)) * np.cos(w * t),
            "rastgele yürüyüş": np.cumsum(rng.normal(size=(R, T)), 1),
            "rastgele sabit X_t = A": rng.normal(size=(R, 1)) * np.ones(T),
        }
        e = rng.normal(size=(R, T + 200))
        a = np.zeros_like(e)
        for i in range(1, T + 200):
            a[:, i] = 0.8 * a[:, i - 1] + e[:, i]
        procs["AR(1), φ = 0.8"] = a[:, 200:]
        m, K = need(ns, "ensemble_stats")(procs["rastgele genlikli kosinüs"])
        close(K[5, 9], np.mean(procs["rastgele genlikli kosinüs"][:, 5] * procs["rastgele genlikli kosinüs"][:, 9]) - m[5] * m[9], rtol=1e-9, what="K tanımı")
        rows, gaps = [], {}
        for name, X in procs.items():
            g = need(ns, "wss_gap")(X)
            tm = need(ns, "time_mean")(X)
            gaps[name] = g
            rows.append([name, f"{g[0]:.3f}", f"{g[1]:.3f}", "evet" if max(g) < 0.15 else "hayır", f"{np.std(tm):.3f}"])
        ok = {k: max(v) < 0.15 for k, v in gaps.items()}
        ensure(ok == {"rastgele fazlı kosinüs": True, "rastgele genlikli kosinüs": False, "rastgele yürüyüş": False, "rastgele sabit X_t = A": True, "AR(1), φ = 0.8": True}, f"WSS sınıflandırması {ok}")
        R_.table(f"{R} gerçekleşme × {T} adım", ["süreç", "ortalama sapması", "kovaryans sapması", "WSS?", "zaman ortalamalarının std'si"], rows,
                 note="Rastgele genlikli kosinüsün ortalaması sabit (0) ama K(t₁, t₂) = cos ωt₁ cos ωt₂ fark yerine mutlak zamana bağlı → WSS değil. Rastgele yürüyüşün varyansı t ile büyür. 'X_t = A' WSS'dir ama ergodik değildir: tek bir gerçekleşmenin zaman ortalaması A'dır, topluluk ortalaması 0 — tek seriden istatistik öğrenen her model (ARMA, RNN) örtük olarak ergodikliğe güvenir.")
    R_.test("durağanlık ve ergodiklik", t)
