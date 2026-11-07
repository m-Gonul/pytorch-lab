NumPy'da iki iş, ikisi de **döngüsüz**:

- `gaussian_mle(X)` — `X: (n, d)`; her sütun için `(mu: (d,), var: (d,))`, MLE varyansı (`ddof=0`).
- `grid_search_p(ys, grid)` — `grid: (m,)` olası $p$ değerleri. Log-likelihood'u **tüm ızgara için tek ifadede** hesapla: `ys` `(n,)`, `grid[:, None]` `(m, 1)` → [[broadcasting]] ile `(m, n)` matris, satır toplamı `(m,)`. En iyi $p$'yi döndür. Izgarada 0 ve 1 olmayacak.
- `class_stats(df, by, col)` — pandas: `by` sütununa göre gruplayıp `col` için `count`, `mean`, `var` (MLE, `ddof=0`) sütunlarından oluşan bir `DataFrame`. İpucu: `groupby(...).agg(...)` ya da `.agg(var=(col, lambda s: s.var(ddof=0)))`.
