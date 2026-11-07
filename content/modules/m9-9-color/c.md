NumPy, **döngü yok**, `uint8` RGB → `uint8`, OpenCV ile **birebir**.

**`rgb2hsv8(img)`** (`color_hsv`, 12 bitlik sabit nokta):

- `v = max`, `vmin = min`, `diff = v − vmin`.
- Tablolar: `sdiv[i] = round((255 << 12) / i)`, `hdiv[i] = round((180 << 12) / (6 i))` ($i>0$; $i=0$ için 0).
- `s = (diff · sdiv[v] + 2048) >> 12`.
- `h = g − b` (v = r ise), değilse `b − r + 2·diff` (v = g ise), değilse `r − g + 4·diff`; `h = (h · hdiv[diff] + 2048) >> 12`; negatifse `+180`.

**`rgb2ycrcb8(img)`** (14 bitlik sabit nokta): `Y = (4899 R + 9617 G + 1868 B + 8192) >> 14`; `δ = (128 << 14) + 8192`; `Cr = ((R − Y)·11682 + δ) >> 14`; `Cb = ((B − Y)·9241 + δ) >> 14`; 0–255'e kırp. (11682 = 0.713·2¹⁴, 9241 = 0.564·2¹⁴.)

**`rgb2hsi(img)`** — b'nin vektörel hali, `float64` `(H, W, 3)`: H derece, S, I.

`>>` NumPy'da negatif sayılarda da aritmetik kaydırmadır (C'deki gibi), `int64` kullan.
