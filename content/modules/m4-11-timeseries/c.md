NumPy.

- `acf(y, maxlag)` — örnek otokorelasyon (ortalama çıkarılmış, $\hat\gamma_k=\frac1n\sum(y_t-\bar y)(y_{t+k}-\bar y)$, $\hat\gamma_k/\hat\gamma_0$); döngüsüz.
- `lag_matrix(y, p)` → `(X, t)`: satır t için `[y_{t-1}, …, y_{t-p}]`, hedef `y_t`; döngüsüz.
- `fit_ar_ls(y, p)` (en küçük kareler) ve `fit_ar_yw(y, p)` (Toeplitz $R\phi=r$); döngüsüz.
- `pacf(y, maxlag)` — gecikme k'daki kısmi otokorelasyon = AR(k) Yule–Walker uyumunun son katsayısı (tek döngü).
- `forecast_ar(y, phi, h)` — yinelemeli h adım (tek döngü).
