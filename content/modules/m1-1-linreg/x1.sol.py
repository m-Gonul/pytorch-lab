import numpy as np


def phi(x, deg):
    """Vandermonde tasarım matrisi: (n,) → (n, deg+1); sütunlar 1, x, …, x**deg."""
    return x[:, None] ** np.arange(deg + 1)


def fit_poly(x, y, deg):
    """En küçük kareler: phi ile lstsq → (deg+1,) katsayılar."""
    c, *_ = np.linalg.lstsq(phi(x, deg), y, rcond=None)
    return c


def rmse(y, y_hat):
    """√(mean((y_hat − y)²)) → float."""
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))


def resid(x, y, deg):
    """Uydurmanın artıkları: r = ŷ − y (l4 deneyindeki işaret)."""
    return phi(x, deg) @ fit_poly(x, y, deg) - y


def quad_sig(x, r):
    """Artıkları MERKEZLENMİŞ x²'ye uyduran katsayı: U deseninin nicel imzası."""
    z = x**2 - (x**2).mean()
    return float(z @ r / (z @ z))


rng = np.random.default_rng(0)
x = np.linspace(-2, 2, 40)
y = 3 * x + 0.8 * x**2 - 1 + 0.4 * rng.normal(size=40)

t1 = fit_poly(x, y, 1)
t2 = fit_poly(x, y, 2)
print(t1)                                                # [0.097, 3.031]: eğriyi ememiyor
print(t2)                                                # [-1.119, 3.031, 0.868]
print(rmse(y, phi(x, 1) @ t1), rmse(y, phi(x, 2) @ t2))   # 1.128 → 0.301
sig1 = quad_sig(x, resid(x, y, 1))
sig2 = quad_sig(x, resid(x, y, 2))
print(sig1, sig2)                                        # −0.868, ~−2e−17
R2 = 1 - rmse(y, phi(x, 1) @ t1) ** 2 / np.var(y)
print("R2 (doğrusal):", R2)                              # ≈ 0.91: yüksek ama biçim yanlış
