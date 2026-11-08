import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        d = 5
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        lam = np.array([4.0, 2.0, 1.0, 0.3, 0.05])
        X = rng.normal(size=(20000, d)) * np.sqrt(lam) @ Q.T
        S = X.T @ X / len(X)
        Wst = need(ns, "dae_optimal")(S, 0.5)
        W = need(ns, "train_linear_dae")(X, 0.5, 0.1, 1500, 0)
        close(W, Wst, atol=0.03, what="SGD ile eğitilen W ≈ kapalı form")
        close(ns["dae_optimal"](S, 0.0), np.eye(d), atol=1e-9, what="gürültüsüz aşırı tam AE: özdeşlik")
        ev = np.linalg.eigvalsh(Wst)[::-1]
        le = np.linalg.eigvalsh(S)[::-1]
        rows = [[f"{l:.2f}", f"{e:.3f}", f"{l / (l + 0.5):.3f}"] for l, e in zip(le, ev)]
        close(ev, le / (le + 0.5), atol=1e-6, what="özdeğer büzülmesi λ/(λ + σ²)")
        R_.table("Doğrusal DAE (σ² = 0.5): W*'ın özdeğerleri", ["veri özdeğeri λ", "W* özdeğeri", "λ/(λ + σ²)"], rows,
                 note="Kod girdiyle aynı boyutta (aşırı tam) ve gürültüsüz bir doğrusal AE özdeşliği öğrenir — işe yaramaz. Gürültü eklenince çözüm, verinin her ana yönünü λ/(λ + σ²) kadar büzen bir Wiener filtresidir: yüksek varyanslı yönler korunur, gürültü düzeyinin altındakiler bastırılır — PCA'nın 'yumuşak' sürümü. Ridge regresyonla aynı matematik.")
    R_.test("doğrusal DAE", t)
