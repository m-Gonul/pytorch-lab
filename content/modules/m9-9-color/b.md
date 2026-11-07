Saf Python, tek piksel fonksiyonları, değerler $[0,1]$:

- `rgb_to_hsi(r, g, b)` → `(H, S, I)`, $H$ derece $[0,360)$. Gri piksel ($R=G=B$) için $H=0$, siyah için $S=0$.
- `hsi_to_rgb(h, s, i)` — üç bölgeli formül.
- `rgb_to_cmyk(r, g, b)` ve `cmyk_to_rgb(c, m, y, k)`.

Kontrol, rastgele renklerde gidiş-dönüş hatasını ölçer.
