import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("q_sample döngüsüz", lambda: max_loops(src, 0, func="q_sample"))
    R.test("ddpm_sample: tek döngü (zaman)", lambda: max_loops(src, 1, func="ddpm_sample", why="örnekler paralel"))
    R.test("ddim_sample: tek döngü (zaman)", lambda: max_loops(src, 1, func="ddim_sample", why="örnekler paralel"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    betas = np.linspace(1e-4, 0.05, 300)
    abar = np.cumprod(1 - betas)
    rng = np.random.default_rng(0)

    def qs():
        x0 = rng.normal(size=(6, 3)); t = np.array([0, 5, 299, 5, 50, 1]); e = rng.normal(size=(6, 3))
        want = np.stack([np.sqrt(abar[ti]) * x + np.sqrt(1 - abar[ti]) * ei for x, ti, ei in zip(x0, t, e)])
        close(need(ns, "q_sample")(x0, t, e, abar), want)
    R.test("q_sample: örnek başına farklı t", qs)
    mu, s = np.array([2.0, -1.0]), np.array([0.5, 1.5])
    oracle = lambda x, t: np.sqrt(1 - abar[t]) * (x - np.sqrt(abar[t]) * mu) / (abar[t] * s ** 2 + 1 - abar[t])

    def dd():
        x = need(ns, "ddpm_sample")(oracle, betas, 6000, 2, np.random.default_rng(1))
        ensure(x.shape == (6000, 2), "şekil (n, d)")
        ensure(np.abs(x.mean(0) - mu).max() < 0.06 and np.abs(x.std(0) - s).max() < 0.06, f"ortalama {x.mean(0).round(3)}, std {x.std(0).round(3)}")
        return f"ortalama {x.mean(0).round(3)}, std {x.std(0).round(3)} (hedef {mu}, {s})"
    R.test("DDPM (300 adım) kâhinle", dd)

    def di():
        rows = []
        for st in (10, 20, 50, 100):
            x = need(ns, "ddim_sample")(oracle, betas, 6000, 2, np.random.default_rng(1), st)
            rows.append([str(st), str(x.mean(0).round(3)), str(x.std(0).round(3))])
        ensure(np.abs(x.mean(0) - mu).max() < 0.05 and np.abs(x.std(0) - s).max() < 0.05, f"100 adım: ortalama {x.mean(0).round(3)}, std {x.std(0).round(3)}")
        y = ns["ddim_sample"](oracle, betas, 6000, 2, np.random.default_rng(1), 100)
        ensure(np.array_equal(x, y), "aynı tohumla aynı sonuç (deterministik)")
        R.table(f"DDIM, kâhin; hedef ortalama {mu}, std {s}", ["adım", "ortalama", "std"], rows,
                note="DDIM birinci mertebe bir ODE çözücüsü gibi davranır: az adımda varyans eksik kalır. DPM-Solver gibi yüksek mertebeli çözücüler bu yüzden var.")
    R.test("DDIM kâhinle (adım sayısı etkisi)", di)

    def det():
        calls = []
        def fn(x, t):
            calls.append(t); return oracle(x, t)
        g = np.random.default_rng(2)
        need(ns, "ddim_sample")(fn, betas, 10, 2, g, 5)
        want = np.linspace(299, 0, 5).round().astype(int).tolist()
        ensure([int(c) for c in calls] == want, f"zaman adımları {calls}, beklenen {want}")
    R.test("DDIM zaman adımları", det)
