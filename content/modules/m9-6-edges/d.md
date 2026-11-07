Canny'nin sonucu: gürültülü 1-B basamaklar için en iyi doğrusal dedektör ≈ Gauss türevi. Bunu öğrenerek görüyoruz.

- `dog_kernel(sigma, n)` — korelasyon (yani `conv1d`) çekirdeği olarak $k(x)=\frac{x}{\sigma^2}e^{-x^2/2\sigma^2}$, $x=-n/2,…,n/2$ (tamsayı).
- `learn_edge_filter(n=21, steps=1500, noise=0.3, seed=0)` — `make_batch` (hazır) gürültülü parça parça sabit sinyaller ve hedef olarak basamak noktalarında yumuşak tepeler verir. Sıfırdan başlayan uzunluğu `n` olan tek bir filtre `w`'yu `F.conv1d(x, w, padding=n//2)` çıkışının hedefe MSE'si (kenarlardan `n` örnek atılarak) ile Adam (lr = 1e-2), 64 × 256'lık yığınlarla öğren; `w`'yu döndür.
- `sobel_torch(img)` — `img (H, W)` float; **tek bir** `conv2d` (2 çıkış kanalı) ve `replicate` dolgusuyla `(gx, gy)`, `cv2.Sobel(…, cv2.CV_64F, …, borderType=cv2.BORDER_REPLICATE)` ile aynı.
