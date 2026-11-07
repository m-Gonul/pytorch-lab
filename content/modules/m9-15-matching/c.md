NumPy, **döngü yok**.

- `ncc_map(img, t)` — `cv2.matchTemplate(img, t, TM_CCOEFF_NORMED)` ile aynı (OpenCV `float32` DFT ve toplamlar kullandığı için neredeyse düz pencerelerde 0.02'ye varan hata yapar; sen `float64` hesapla — kontrol seni doğrudan tanımla, OpenCV'yi yalnız dokulu pencerelerde karşılaştırır). Pay: sıfır ortalamalı şablonla korelasyon (`scipy.signal.fftconvolve(img, tz[::-1, ::-1], "valid")`); payda: pencere varyansı **integral görüntülerden** ($\sum f$ ve $\sum f^2$). Varyans ≈ 0 olan yerde 0.
- `normalize_points(p)` → `(q, T)`: Hartley; $q=T\tilde p$.
- `dlt(src, dst)` — normalize et, $A$ (2N × 9), `np.linalg.svd`, son satır, geri normalize ($H=T_d^{-1}\hat HT_s$), $h_{33}=1$.
- `ransac(src, dst, thr=3.0, iters=500, seed=0)` → `(H, mask)`: `rng = np.random.default_rng(seed)`, örnekler `rng.random((iters, N)).argsort(1)[:, :4]`; **tüm hipotezleri tek seferde** (yığın `(iters, 8, 9)` → `np.linalg.svd` yığın üzerinde çalışır); hepsinin yeniden izdüşüm hatası `(iters, N)`; en çok içerdeğerli (eşitlikte ilk); `dlt` ile içerdeğerlerden yeniden kestir; son maske bu $H$'ye göre.
