Listelerle üç fonksiyon yaz; `numpy`, `torch` ve `math.dist`/`math.hypot` gibi kısayollar yok, ama `math.sqrt` serbest.

- `dot(u, v)` — uzunluklar farklıysa `ValueError`.
- `norm(u)` — `dot`'u kullanarak.
- `cosine(u, v)` — sıfır vektör için tanımsız; bu durumda `ValueError` fırlat (`if` burada işe yarıyor).
