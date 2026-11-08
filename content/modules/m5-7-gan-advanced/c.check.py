import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüler", lambda: (max_loops(src, 0, func="sqrtm_psd"), max_loops(src, 0, func="frechet"), max_loops(src, 1, func="sqrtm_ns")))
    rng = np.random.default_rng(0)

    def t():
        A = rng.normal(size=(6, 6))
        S = A @ A.T + 0.1 * np.eye(6)
        R = need(ns, "sqrtm_psd")(S)
        close(R @ R, S, atol=1e-9, what="√S · √S = S")
        close(need(ns, "sqrtm_ns")(S, 60), R, atol=1e-6, what="Newton–Schulz")
        close(need(ns, "frechet")(np.array([1.0, 0]), np.eye(2), np.zeros(2), np.diag([4.0, 9])), 6.0, what="el örneği")
        S2 = (lambda B: B @ B.T)(rng.normal(size=(6, 6)))
        w = np.linalg.eigvals(S @ S2)
        close(ns["frechet"](np.zeros(6), S, np.zeros(6), S2), np.trace(S) + np.trace(S2) - 2 * np.sum(np.sqrt(w.real.clip(0))), atol=1e-6, what="tr√(S₁S₂) simetrik biçimle aynı")
        P = np.eye(4)[rng.integers(0, 4, 4000)]
        close(need(ns, "inception_score")(P, 1)[0], 4.0, atol=0.02, what="dengeli ve emin → C")
        centers = rng.normal(0, 3, (10, 16))
        real = centers[rng.integers(0, 10, 5000)] + rng.normal(size=(5000, 16))
        rows = []
        for name, fake in (("gerçekle aynı dağılım", centers[rng.integers(0, 10, 5000)] + rng.normal(size=(5000, 16))),
                           ("sınıf içi std 0.5", centers[rng.integers(0, 10, 5000)] + 0.5 * rng.normal(size=(5000, 16))),
                           ("5 mod (yarısı düştü)", centers[rng.integers(0, 5, 5000)] + rng.normal(size=(5000, 16))),
                           ("1 mod", centers[np.zeros(5000, int)] + rng.normal(size=(5000, 16))),
                           ("ortalama 0.5 kaymış", centers[rng.integers(0, 10, 5000)] + 0.5 + rng.normal(size=(5000, 16)))):
            rows.append([name, f"{need(ns, 'fid')(real, fake):.2f}"])
        f = [float(r[1]) for r in rows]
        ensure(f[0] < 0.2 and f[0] < f[4] < f[3] and f[2] < f[3] and f[1] > f[0], f"FID sırası {f}")
        R_.table("16 boyutlu öznitelik uzayında 10 modlu 'gerçek' veri", ["üretilen", "FID"], rows,
                 note="FID tek bir Gauss uydurur: mod düşürmek ortalamayı ve kovaryansı değiştirdiği için yakalanır, ama çok modluluğun kendisini değil yalnız ilk iki momentini görür. Aynı dağılımdan çekilen iki küme bile > 0 verir (sonlu örneklem yanlılığı, x2).")
    R_.test("matris karekökü ve FID", t)
