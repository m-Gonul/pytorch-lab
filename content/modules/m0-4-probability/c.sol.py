import numpy as np
import pandas as pd


def gaussian_mle(X):
    """X: (n, d) → (mu: (d,), var: (d,)). Döngü yok."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    return mu, ((X - mu) ** 2).mean(axis=0)


def grid_search_p(ys, grid):
    """En yüksek Bernoulli log-likelihood'u veren p. Döngü yok."""
    ys = np.asarray(ys, dtype=float)
    P = np.asarray(grid, dtype=float)[:, None]                  # (m, 1)
    ll = (ys * np.log(P) + (1 - ys) * np.log(1 - P)).sum(1)    # (m,)
    return float(grid[np.argmax(ll)])


def class_stats(df, by, col):
    """by'a göre grupla; col için count, mean, var (ddof=0) sütunları."""
    return df.groupby(by)[col].agg(count="count", mean="mean", var=lambda s: s.var(ddof=0))


print(gaussian_mle(np.array([[2.0, 1.0], [4.0, 1.0], [6.0, 4.0]])))
print(grid_search_p(np.array([1, 0, 1, 1]), np.linspace(0.01, 0.99, 99)))   # 0.75
df = pd.DataFrame({"sinif": ["a", "a", "b", "b", "b"], "boy": [1.0, 3.0, 2.0, 4.0, 6.0]})
print(class_stats(df, "sinif", "boy"))
