Tüm $(j,t)$ adaylarını **döngüsüz** değerlendir:

1. `order = np.argsort(X, axis=0, kind="stable")` — her sütunu ayrı sırala; `Xs = np.take_along_axis(X, order, 0)`.
2. Sıralı etiketlerin one-hot'ı `(n, d, C)`; `cumsum(axis=0)` → ilk $i+1$ örneğin sınıf sayımları = sol çocuk. Sağ = toplam − sol.
3. Her pozisyon için ağırlıklı Gini `(n-1, d)`.
4. $X_s[i]=X_s[i+1]$ olan pozisyonlar geçersiz (`np.inf`).
5. Eşitlik kuralı b ile aynı olsun: en küçük skor, sonra küçük özellik, sonra küçük eşik.

- `best_split_np(X, y, C)` → `(j, t, skor)` ya da `None`.
