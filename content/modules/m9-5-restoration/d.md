Küçük bir artık (residual) gürültü giderici:

- `Denoiser(c=32, depth=4)` — `Conv3×3 → ReLU → (Conv3×3 → ReLU) × (depth-2) → Conv3×3`, hepsi `padding=1`; ağ **gürültüyü** tahmin eder, çıkış `x - net(x)`.
- `train_denoiser(img, sigma, steps=500, batch=16, patch=48, seed=0)` — `img` [0,1] float tensör; her adımda rastgele yamalar, $\sigma$ Gauss gürültüsü ekle, MSE (Adam, lr = 1e-3). Modeli döndür.

Kontrol, `camera` ile eğitip görülmemiş `coins` görüntüsünde $\sigma=25/255$ için PSNR'ı Gauss bulanıklaştırma ile karşılaştırır.
