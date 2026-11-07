import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü kuralları", lambda: (max_loops(src, 0, func="early_stop"), max_loops(src, 0, func="standardize_ok")))

    def t():
        f = need(ns, "kfold_indices")(103, 5, 0)
        ensure(len(f) == 5 and sorted(np.concatenate([v for _, v in f]).tolist()) == list(range(103)), "katlar ayrık ve tam")
        ensure(all(len(np.intersect1d(a, b)) == 0 for a, b in f), "eğitim/doğrulama ayrık")
        ensure(need(ns, "early_stop")([1.0, 0.8, 0.7, 0.72, 0.75, 0.71, 0.9], 2) == (4, 2), "erken durdurma (4. epoch'ta dur, 2. epoch en iyi)")
        ensure(ns["early_stop"]([3.0, 2, 1, 0.5], 2) == (3, 3), "hiç durmazsa son epoch")
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, 60)
        y = np.sin(3 * x) + 0.2 * rng.normal(size=60)
        cv = need(ns, "poly_cv")(x, y, [1, 3, 5, 9, 15])
        best = min(cv, key=cv.get)
        ensure(3 <= best <= 9 and cv[15] > cv[best], "orta derece seçilmeli")
        Xtr = rng.normal(0, 1, (50, 3))
        Xte = rng.normal(3, 1, (50, 3))
        a, _ = need(ns, "standardize_leak")(Xtr, Xte)
        b, _ = need(ns, "standardize_ok")(Xtr, Xte)
        close(b.mean(0), np.zeros(3), atol=1e-9, what="eğitimde ortalama 0")
        ensure(abs(a.mean()) > 0.5, "sızıntılı ölçek eğitim dağılımını kaydırır")
        R_.table("Polinom derecesi, 5 katlı CV (n = 60)", ["derece", "CV MSE"], [[d, f"{v:.3f}"] for d, v in cv.items()],
                 note="CV hatası klasik U çizer. Seçilen dereceyle tüm eğitim verisinde yeniden eğit, testte bir kez değerlendir.")
    R_.test("CV, erken durdurma, ölçekleme", t)
