import numpy as np
import pandas as pd


def gaussian_mle(X):
    """X: (n, d) → (mu: (d,), var: (d,)). Döngü yok."""
    ...


def grid_search_p(ys, grid):
    """En yüksek Bernoulli log-likelihood'u veren p. Döngü yok."""
    ...


def class_stats(df, by, col):
    """by'a göre grupla; col için count, mean, var (ddof=0) sütunları."""
    ...


print(gaussian_mle(np.array([[2.0, 1.0], [4.0, 1.0], [6.0, 4.0]])))
print(grid_search_p(np.array([1, 0, 1, 1]), np.linspace(0.01, 0.99, 99)))   # 0.75
df = pd.DataFrame({"sinif": ["a", "a", "b", "b", "b"], "boy": [1.0, 3.0, 2.0, 4.0, 6.0]})
print(class_stats(df, "sinif", "boy"))
