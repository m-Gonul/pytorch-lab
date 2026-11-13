import numpy as np


def soft(z, t):
    """Yumuşak eşikleme: sign(z)·max(|z| − t, 0); |z| ≤ t → tam 0."""
    ...


def enet_cd(X, y, lam1, lam2, iters=200, w0=None):
    """Elastic net koordinat inişi: (1/2n)‖y−Xw‖² + λ₁‖w‖₁ + λ₂/2‖w‖². → (d,)"""
    ...


def enet_path(X, y, lams, lam2=0.5):
    """λ yolu: lams büyükten küçüğe, sıcak başlangıçla. → (L, d)."""
    ...


def grouping_gap(w):
    """Özdeş ikilinin ağırlık farkı |w₁ − w₂| (gruplama etkisinin ölçüsü)."""
    ...


rng = np.random.default_rng(0)
n, d = 40, 6
x1 = rng.normal(size=n)
x2 = x1 + 1e-8 * rng.normal(size=n)  # özdeş ikili (neredeyse)
X = np.column_stack([x1, x2, rng.normal(size=(n, 4))])
w_true = np.array([1.5, 1.5, -2.0, 0.0, 0.0, 0.0])
y = X @ w_true + 0.3 * rng.normal(size=n)

lams = np.logspace(0.5, -3, 31)        # 3.16 → 0.001, büyükten küçüğe
W = enet_path(X, y, lams, lam2=0.5)
wA = W[15]                             # orta λ
print(wA)                              # ilk üç girer, son üç ~0
print("gap:", grouping_gap(wA))        # < 1e-6: özdeş ikili eşit paylaştırır
wL = enet_cd(X, y, 0.1, 0.0, iters=400)  # saf lasso (λ₂ = 0), aynı λ civarı
print(wL)                              # ikiliden birini seçer, diğerini tam 0 yapar
print("λ_max:", np.abs(X.T @ y / n).max())  # ≈ 2.27: üstünde hepsi tam 0
