import numpy as np


def soft(z, t):
    """Yumuşak eşikleme: sign(z)·max(|z| − t, 0); |z| ≤ t → tam 0."""
    return float(np.sign(z) * np.maximum(np.abs(z) - t, 0.0))


def enet_cd(X, y, lam1, lam2, iters=200, w0=None):
    """Elastic net koordinat inişi: (1/2n)‖y−Xw‖² + λ₁‖w‖₁ + λ₂/2‖w‖². → (d,)"""
    n, d = X.shape
    w = np.zeros(d) if w0 is None else np.array(w0, dtype=float)
    col_sq = (X**2).mean(axis=0)
    for _ in range(iters):
        for j in range(d):
            r = y - X @ w + X[:, j] * w[j]  # j'yi kapsayan artık
            z = X[:, j] @ r / n
            w[j] = soft(z, lam1) / (col_sq[j] + lam2)
    return w


def enet_path(X, y, lams, lam2=0.5):
    """λ yolu: lams büyükten küçüğe, sıcak başlangıçla. → (L, d)."""
    W = np.empty((len(lams), X.shape[1]))
    w = None
    for i, lam in enumerate(lams):
        w = enet_cd(X, y, lam, lam2, iters=60 if w is not None else 200, w0=w)
        W[i] = w
    return W


def grouping_gap(w):
    """Özdeş ikilinin ağırlık farkı |w₁ − w₂| (gruplama etkisinin ölçüsü)."""
    return float(abs(w[0] - w[1]))


rng = np.random.default_rng(0)
n, d = 40, 6
x1 = rng.normal(size=n)
x2 = x1 + 1e-8 * rng.normal(size=n)  # özdeş ikili (neredeyse)
X = np.column_stack([x1, x2, rng.normal(size=(n, 4))])
w_true = np.array([1.5, 1.5, -2.0, 0.0, 0.0, 0.0])
y = X @ w_true + 0.3 * rng.normal(size=n)

lams = np.logspace(0.5, -3, 31)        # 3.16 → 0.001, büyükten küçüğe
W = enet_path(X, y, lams, lam2=0.5)
wA = W[15]
print(wA)                              # ≈ [1.08, 1.08, −1.37, −0.05, −0.13, −0.07]
print("gap:", grouping_gap(wA))        # < 1e-6: özdeş ikili eşit paylaştırır
wL = enet_cd(X, y, 0.1, 0.0, iters=400)  # saf lasso (λ₂ = 0)
print(wL)                              # ≈ [2.83, 0, −2.02, 0, ~0, ~0]: ikiliyi bölemedi
print("λ_max:", np.abs(X.T @ y / n).max())  # ≈ 2.27
