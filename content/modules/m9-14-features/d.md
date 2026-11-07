DoG piramidi ve **ölçek seçimi**.

- `dog_pyramid(img, octaves=4, s=3, sigma0=1.6)` — `img (H, W)` `float64` tensör. Her oktavda $s+3$ Gauss görüntüsü: ilki tabanın $\sigma_0$ ile bulanığı (ilk oktavda görüntünün kendisi), sonrakiler artımlı $\sqrt{\sigma_i^2-\sigma_{i-1}^2}$, $\sigma_i=\sigma_0k^i$. Gauss: yarıçap $\lceil4\sigma\rceil$, toplamı 1, `replicate`, ayrılabilir `conv2d`. DoG: `(s+2, H_o, W_o)` tensör. Sonraki oktavın tabanı: bu oktavın $s$. görüntüsü `[::2, ::2]`. Liste döndür.
- `detect(img, octaves=4, s=3, sigma0=1.6, thr=0.03)` — her oktavda 3 × 3 × 3 yerel en büyük ya da en küçük (`max_pool3d`, eşitlik dahil), $|D|>$ `thr`, ilk ve son DoG katmanı ve 1 piksellik kenar hariç. Anahtar nokta `(x·2^o, y·2^o, σ_0·2^{o+i/s}, D)` listesi.

Kontrol, yarıçapları farklı Gauss bloblarında bulunan ölçeğin blob boyutuyla **orantılı** çıktığını ve DoG yanıtının ölçekten bağımsız kaldığını gösterir.
