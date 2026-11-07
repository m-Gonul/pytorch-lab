Üç sürüm yaz; kontrol her birinin döngü derinliğini ayrı ayrı denetler:

- `sqdist_one_loop(Xte, Xtr)` — test satırları üzerinde **tek** döngü; her satır için `((Xtr - x) ** 2).sum(1)` (broadcasting).
- `sqdist_no_loop(Xte, Xtr)` — **döngü yok**; açılım formülü. Negatif çıkan küçük değerleri `np.maximum(D, 0)` ile kırp.
- `knn_predict_np(D, ytr, k)` — **döngü yok**. Sıralama için `np.argsort(D, axis=1, kind="stable")` (eşit mesafede küçük indeks önce gelir). Oylar: komşu etiketlerini one-hot'a çevirip topla — `np.eye(C)[ytr[idx]]` shape'i `(n_te, k, C)`, `sum(axis=1)` sonrası `argmax` (eşitlikte ilk, yani küçük etiket).
