**Grup sızıntısı**, saf Python. Her "denek" birkaç gürültülü ölçüm üretir (aynı kişinin tekrarlanan örnekleri).

- `make_groups(n_groups, per_group, seed)` → `(X, y, g)`: grup başına rastgele bir merkez ve etiket (etiket merkezle **zayıf** ilişkili), grup içi örnekler merkez + küçük gürültü.
- `nn1_predict(Xtr, ytr, Xte)` — 1-en yakın komşu.
- `split_random(n, frac, seed)` ve `split_groups(g, frac, seed)` → `(train_idx, test_idx)`.
- `accuracy_both(X, y, g, seed)` → `(rastgele bölme doğruluğu, grup bölme doğruluğu)`.
