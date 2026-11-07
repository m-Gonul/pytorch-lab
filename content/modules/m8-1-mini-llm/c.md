NumPy, **döngü yok**. Token kimlikleri: `0..9` rakamlar, `10` = `+`, `11` = `=`, `12` = satır sonu.

- `make_dataset(n, rng)` → `(n, 10)` int dizi; `a, b = rng.integers(0, 100, (2, n))`; rakamları `//` ve `%` ile çıkar, string üretme.
- `encode(text)` → `(satır, 10)`; arama tablosu: `lut[np.frombuffer(text.encode(), np.uint8)]`.
- `answer_entropy(D)` → cevap konumlarının (6, 7, 8) marjinal dağılımlarının entropileri toplamı (nat). Model bundan iyi olmalı; "sabit tahmin" tabanı budur. `np.bincount` ile.
