import numpy as np
import pandas as pd


def normal_eq(X, y):
    """→ (w: (d,), b: float). Döngü yok."""
    X = np.asarray(X, dtype=float)
    Xt = np.column_stack([X, np.ones(len(X))])
    theta, *_ = np.linalg.lstsq(Xt, np.asarray(y, dtype=float), rcond=None)
    return theta[:-1], float(theta[-1])


def fit_gd(X, y, lr=0.1, steps=500):
    """→ (w, b, losses). Yalnız adım döngüsü."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    losses = []
    for _ in range(steps):
        e = X @ w + b - y
        losses.append(float(e @ e / n))
        w = w - lr * (2 / n) * (X.T @ e)
        b = b - lr * (2 / n) * e.sum()
    return w, b, np.array(losses)


def fit_df(df, target):
    """→ (w: pd.Series [özellik isimleriyle], b)."""
    feats = df.drop(columns=[target])
    w, b = normal_eq(feats.to_numpy(), df[target].to_numpy())
    return pd.Series(w, index=feats.columns), b


rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3))
y = X @ np.array([2.0, -1.0, 0.5]) + 3 + 0.1 * rng.normal(size=100)
print(normal_eq(X, y))
df = pd.DataFrame(X, columns=["alan", "yas", "kat"]).assign(fiyat=y)
print(fit_df(df, "fiyat"))
