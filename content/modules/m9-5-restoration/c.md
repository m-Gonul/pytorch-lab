NumPy, **döngü yok**. Kayan pencere: `np.lib.stride_tricks.sliding_window_view(np.pad(img, r, mode="edge"), (k, k))` → `(H, W, k, k)`.

- `median_np(img, k)` — `cv2.medianBlur` ile birebir.
- `adaptive_mean(img_u8, noise_var, k=5)` — görüntüyü $[0,1]$'e normalize et (`float32`), yerel ortalama/varyans (kopyalama kenarı), oran $\min(\sigma_\eta^2/\sigma_S^2, 1)$, sonucu `uint8`'e geri çevir (`np.rint`, kırp).
- `psnr(a, b)` — `cv2.PSNR` ile aynı. OpenCV sıfıra bölmeyi önlemek için $20\log_{10}\frac{255}{\sqrt{\text{MSE}}+\varepsilon}$ hesaplar ($\varepsilon$ = `np.finfo(np.float64).eps`); özdeş görüntülerde sonuç ≈ 361.2 dB.
