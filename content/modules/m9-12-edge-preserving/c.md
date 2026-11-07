`clahe_cv(img, clip=2.0, tiles=(8, 8))` — `cv2.createCLAHE(clip, tiles).apply(img)` ile **birebir** (`tiles = (tilesX, tilesY)`), NumPy, **döngü yok**:

1. Dolgu kuralı (giriş metnine bak), döşeme boyutu $t_h\times t_w$, $N=t_ht_w$.
2. Tüm döşemelerin histogramı tek seferde: döşemeleri `reshape/transpose` ile `(T, N)` yap, `np.add.at` ya da ofsetli `bincount`.
3. Kırpma ve yeniden dağıtma vektörel: kalan dağıtımı `(i % step == 0) & (i // step < r)` maskesiyle.
4. LUT: `float32` `cdf · (255/N)`, `np.rint`.
5. Enterpolasyon `float32`'de: $t_x=x\cdot(1/t_w)-0.5$, $x_1=\lfloor t_x\rfloor$, $a=t_x-x_1$, döşeme indisleri $[0,T_x-1]$'e kırpılır; dört LUT bakışı `lut[ty, tx, v]`; sonuç `np.rint`.
