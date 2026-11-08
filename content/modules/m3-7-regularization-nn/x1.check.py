import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("alpha_dropout", "std_dropout")])

    def t():
        rng = np.random.default_rng(0)
        x = rng.normal(size=400000)
        for p in (0.1, 0.3):
            y = need(ns, "alpha_dropout")(x, p, rng)
            close((y.mean(), y.var()), (0.0, 1.0), atol=0.01, what=f"alpha dropout p = {p}: ortalama 0, varyans 1")
            z = need(ns, "std_dropout")(x, p, rng)
            close(z.mean(), 0.0, atol=0.01, what="standart dropout ortalamayı korur")
            ensure(z.var() > 1 + p / (1 - p) - 0.05, "… ama varyansı 1/(1 − p) katına çıkarır")
        lam, al = 1.0507009873554805, 1.6732632423543772
        selu = lambda z: lam * np.where(z > 0, z, al * (np.exp(np.minimum(z, 0)) - 1))
        rows = []
        for name, drop in (("dropout yok", None), ("standart dropout", ns["std_dropout"]), ("alpha dropout", ns["alpha_dropout"])):
            H = rng.normal(size=(2000, 128))
            for _ in range(30):
                H = selu(H @ (rng.normal(size=(128, 128)) / np.sqrt(128)))
                if drop is not None:
                    H = drop(H, 0.2, rng)
            rows.append([name, f"{H.mean():+.3f}", f"{H.std():.3f}"])
        ensure(abs(float(rows[2][2]) - 1) < 0.15 and abs(float(rows[1][2]) - 1) > abs(float(rows[2][2]) - 1), "alpha dropout öz-normalizasyonu korur")
        R_.table("SELU + LeCun, 30 katman, her katmandan sonra p = 0.2 bırakma", ["bırakma", "son ortalama", "son std"], rows,
                 note="Standart dropout'un 0 değeri SELU için 'nötr' değildir (SELU'nun negatif doygunluğu −1.758) ve 1/q ölçeklemesi varyansı şişirir; birikince katmanlar sabit noktadan uzaklaşır. Alpha dropout bırakılan birimleri doygunluk değerine kilitler ve afin düzeltmeyle (0, 1) momentlerini korur.")
    R_.test("alpha dropout", t)
