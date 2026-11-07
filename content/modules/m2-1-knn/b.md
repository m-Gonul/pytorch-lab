Listelerle iki fonksiyon:

- `sqdist_loops(Xte, Xtr)` — iki iç içe döngüyle `nte × ntr` kare mesafe listesi.
- `knn_predict(D, ytr, k)` — her test satırı için: komşuları `(mesafe, indeks)` sırasına göre diz, ilk `k`'nın etiketlerini say (`dict` ya da `collections.Counter`), en çok oyu alanı seç; eşitlikte küçük etiket.

`numpy`/`torch` yasak.
