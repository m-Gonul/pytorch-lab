NumPy (`scipy.ndimage` serbest), **döngü yok** (bisection gibi arama döngüleri hariç, aşağıya bak).

**`ssim(x, y, R=255)`** — `skimage.metrics.structural_similarity(x, y, data_range=R, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)` ile **aynı**:

- `float64`; $\mu$, $E[x^2]$, $E[y^2]$, $E[xy]$ = `ndimage.gaussian_filter(…, sigma=1.5, truncate=3.5, mode="reflect")`.
- $v_x=E[x^2]-\mu_x^2$ (normalizasyon 1), SSIM haritası, kenardan `pad = 5` piksel kırpılmış ortalaması.

**Eşit MSE bozulmaları** (Wang 2004, Şekil 2) — her biri `float64` döndürür, hedef MSE'ye çok yakın:

- `mean_shift(img, mse)`, `contrast(img, mse)` ($\mu+a(x-\mu)$, $(a-1)^2\sigma^2=\text{MSE}$).
- `noise(img, mse, seed=0)` — Gauss gürültüsü, MSE'si tam hedefe ölçeklenmiş.
- `impulse(img, mse, seed=0)` — rastgele sırayla pikselleri 0 ya da 255 yap (yazı-tura), kümülatif hata hedefe ulaşana dek (döngüsüz: `cumsum` + `searchsorted`).
- `blur(img, mse)` — `cv2.GaussianBlur(…, (0, 0), σ)`; σ'yı ikiye bölmeyle bul (bir `for` döngüsü serbest).
- `jpeg(img, mse)` — `cv2.imencode(".jpg", …, [cv2.IMWRITE_JPEG_QUALITY, q])`; MSE'si hedefe en yakın kalite (`q` üzerinde bir döngü serbest).
