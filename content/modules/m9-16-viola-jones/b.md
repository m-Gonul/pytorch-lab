Saf Python. Dikdörtgen biçimi scikit-image'inki gibi: `[(r0, c0), (r1, c1)]`, **uçlar dahil**.

- `integral(img)` — `(H+1) × (W+1)` liste, ilk satır ve sütun 0.
- `rect_sum(ii, rect)` — 4 erişim.
- `haar(ii, rects, kind)` — `kind` ∈ `"type-2-x"`, `"type-2-y"`, `"type-3-x"`, `"type-3-y"`, `"type-4"`; giriş metnindeki işaretler.
- `features_at(img, coords, kinds)` — bir görüntü için öznitelik listesi.

Kontrol, `skimage.feature.haar_like_feature` ile aynı değerleri bekler.
