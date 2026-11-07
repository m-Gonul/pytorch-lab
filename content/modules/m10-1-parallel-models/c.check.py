import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü kuralları", lambda: (max_loops(src, 1, func="critical_path"), max_loops(src, 0, func="fit_amdahl"), max_loops(src, 0, func="speedup_grid")))

    def cp():
        rng = np.random.default_rng(0)
        n = 60
        A = np.triu((rng.random((n, n)) < 0.08).astype(float), 1)
        dur = rng.integers(1, 10, n).astype(float)
        ef, S = need(ns, "critical_path")(A, dur)
        ref = np.zeros(n)
        for j in range(n):
            ref[j] = dur[j] + max([ref[i] for i in range(j) if A[i, j]], default=0)
        close(ef, ref, what="en erken bitişler")
        close(S, ref.max(), what="kritik yol")
    R_.test("critical_path", cp)

    def fit():
        p = np.array([1, 2, 4, 8, 16, 32.0])
        T = 10 * (0.07 + 0.93 / p) * (1 + 0.01 * np.random.default_rng(1).normal(size=6))
        s, T1 = need(ns, "fit_amdahl")(p, T)
        ensure(abs(s - 0.07) < 0.01 and abs(T1 - 10) < 0.3, f"seri oran ≈ 0.07 (bulunan {s:.3f})")
        am, gu = need(ns, "speedup_grid")(np.array([0.5, 0.9, 0.99, 0.999]), np.array([4, 16, 64, 1024.0]))
        close(am[1, 0], 1 / (0.1 + 0.9 / 4), what="Amdahl tablosu")
        R_.table("Amdahl hızlanması (Gustafson parantezde)", ["f", "p = 4", "16", "64", "1024"],
                 [[f, *[f"{am[i, j]:.1f} ({gu[i, j]:.0f})" for j in range(4)]] for i, f in enumerate([0.5, 0.9, 0.99, 0.999])],
                 note="Binlerce çekirdekten yararlanmak için seri oran binde birin altında olmalı (Amdahl) — ya da problem çekirdeklerle büyümeli (Gustafson). GPU'lar ikinci dünyanın aracıdır.")
    R_.test("Amdahl uydurma ve tablolar", fit)
