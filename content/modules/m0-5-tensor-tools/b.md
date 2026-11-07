Saf Python'da kendi minik "tensör"ünü yaz: düz bir liste (`data`) + shape + stride + offset.

- `contiguous_strides(shape)` — row-major stride'lar, ör. `(2, 3, 4) → (12, 4, 1)`.
- `get(data, strides, offset, index)` — adres formülüyle tek eleman.
- `transpose(shape, strides, i, j)` → `(yeni_shape, yeni_strides)`; veri kopyalanmaz.
- `materialize(data, shape, strides, offset)` — iç içe liste üret (herhangi bir boyut sayısı; özyineleme ya da döngü).
- `windows(n, w)` → kayan pencere view'u için `(shape, strides)`; `materialize` ile açınca pencereler çıkmalı.

`numpy`/`torch` yasak.
