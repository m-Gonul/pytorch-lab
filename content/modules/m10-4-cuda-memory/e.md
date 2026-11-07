**CPU'da aynı fizik:** önbellek satırı (64 B) GPU sektörünün karşılığı.

- `strided_cost(n, strides, repeat=3)` → her adım için **eleman başına** süre (ns): `x[::s].sum()` (float32, x büyük ve önbelleğe sığmaz).
- `transpose_variants(N)` → sözlük: `x.t().contiguous()`, karo karo (T = 32) `reshape`/`permute` ile devrik kopya, satır satır Python döngüsüyle (`out[:, i] = x[i]`) — süreler.
