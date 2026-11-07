Tek özellikli ($x$ skaler) regresyonu listelerle yaz:

- `gd_step(xs, ys, w, b, lr)` → `(w_yeni, b_yeni, kayıp)`. Kayıp, güncellemeden **önceki** parametrelerle hesaplanır.
- `fit(xs, ys, lr=0.05, steps=2000)` → `(w, b)`; `gd_step`'i tekrar çağırır.
- `closed_form(xs, ys)` → `(w*, b*)`; yukarıdaki ortalama/kovaryans formülü.

Kontrol, `fit` sonucunun `closed_form` ile 4 basamak eşleşmesini bekler. `numpy` ve `torch` yasak.
