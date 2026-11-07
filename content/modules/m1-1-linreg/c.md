Şimdi $d$ özellik. Örnekler üzerindeki döngü kalkıyor; **adım döngüsü kalıyor** (her adım bir öncekine bağlı, paralelleştirilemez). Kontrol, `normal_eq` içinde hiç döngü, `fit_gd` içinde tek katman döngü olmasını bekler.

- `normal_eq(X, y)` → `(w, b)`. `X`'e 1 sütunu ekle, `np.linalg.lstsq` (ya da `solve`) ile çöz. `np.linalg.inv` kullanma: sayısal olarak daha kötü.
- `fit_gd(X, y, lr=0.1, steps=500)` → `(w, b, losses)`; `losses` her adımdaki kayıp.
- `fit_df(df, target)` — pandas `DataFrame`'den `target` sütununu ayırıp `normal_eq` uygular; `w`'yi **özellik isimleriyle indekslenmiş** bir `pd.Series` olarak döndürür. İsimler, modelin hangi katsayının neye ait olduğunu unutmamasını sağlar.
