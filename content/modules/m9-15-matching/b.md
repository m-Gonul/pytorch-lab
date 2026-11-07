Saf Python.

- `ncc(a, b)` — iki düz liste arasında korelasyon katsayısı (paydası 0 ise 0).
- `match_template(img, t)` — "valid" NCC haritası (liste listesi), `cv2.matchTemplate(…, cv2.TM_CCOEFF_NORMED)` ile aynı boyut ve (1e-4 içinde) aynı değerler.
- `solve_h4(src, dst)` — 4 nokta eşinden $h_{33}=1$ alarak 8 × 8 doğrusal sistemi **kısmi pivotlamalı Gauss eliminasyonuyla** çöz; $3\times3$ liste.
- `apply_h(H, x, y)` — homojen koordinatla $(x', y')$.
