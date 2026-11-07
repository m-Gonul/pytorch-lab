Saf Python; shape'ler tuple, 2-boyutlu gradyanlar liste listesi.

- `broadcast_shape(s1, s2)` — NumPy kuralı: sağdan hizala, her eksende eşit ya da biri 1 olmalı, eksik eksen 1 sayılır. Uyumsuzsa `ValueError`.
- `unbroadcast2d(G, shape)` — `G` $(n\times m)$ liste; `shape` şunlardan biri: `(n, m)`, `(m,)`, `(1, m)`, `(n, 1)`, `()`. Dönüş: `(m,)` → liste, `(1, m)` → `[liste]`, `(n, 1)` → `[[s], [s], …]`, `()` → sayı.
