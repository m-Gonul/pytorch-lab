import numpy as np
import pandas as pd


def normal_eq(X, y):
    """→ (w: (d,), b: float). Döngü yok."""
    ...


def fit_gd(X, y, lr=0.1, steps=500):
    """→ (w, b, losses). Yalnız adım döngüsü."""
    ...


def fit_df(df, target):
    """→ (w: pd.Series [özellik isimleriyle], b)."""
    ...


rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3))
y = X @ np.array([2.0, -1.0, 0.5]) + 3 + 0.1 * rng.normal(size=100)
print(normal_eq(X, y))
df = pd.DataFrame(X, columns=["alan", "yas", "kat"]).assign(fiyat=y)
print(fit_df(df, "fiyat"))
