NumPy, **döngü yok**; her dönüşüm önce 256 elemanlı bir LUT kurar, sonra `lut[img]`.

- `equalize(img, variant)` — `"class"` ya da `"opencv"`; `np.bincount(…, minlength=256)` ve `cumsum`.
- `negative(img)`, `log_transform(img)` ($c=255/\ln256$), `gamma(img, g)` ($255\,(r/255)^g$) — çıkışlar `uint8`'e yuvarlanır (`np.rint`).
