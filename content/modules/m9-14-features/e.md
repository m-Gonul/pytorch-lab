`describe(img, kps)` — `img (H, W)` NumPy, `kps (N, 4)` `float64` tensör `[x, y, σ, θ]` → `(N, 128)`; tüm noktalar **aynı anda** (döngü yok):

1. Örnek ızgarası $u,v\in\{-7.5,…,7.5\}$ (16 × 16), adım $\Delta=3\sigma/4$, θ ile döndürülmüş: $p=(x,y)+\Delta R(\theta)(u,v)$.
2. Döndürülmüş çerçevede gradyan: $g_u=I(p+\Delta Re_u)-I(p-\Delta Re_u)$, $g_v$ benzer — `grid_sample` (bilineer, `align_corners=True`, `padding_mode="border"`), dört örnekleme.
3. Büyüklük × $e^{-(u^2+v^2)/(2\cdot8^2)}$, açı `atan2(gv, gu) mod 2π`, 8 bölme (sert, `floor`).
4. Alt bölge $(\lfloor(v+7.5)/4\rfloor,\lfloor(u+7.5)/4\rfloor)$, indis `alt_bölge·8 + bölme`, `scatter_add_`.
5. L2 normalize, 0.2'de kırp, tekrar normalize.

Gradyanlar anahtar noktanın kendi çerçevesinde hesaplandığı için betimleyici **dönmeden bağımsızdır**.
