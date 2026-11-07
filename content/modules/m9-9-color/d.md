`img (B, 3, H, W)` float.

- `di_zenzo(img)` → `(F, theta)`: her kanalda Sobel (gruplu `conv2d`, `groups=3`, `replicate` dolgu), $g_{xx}, g_{yy}, g_{xy}$ kanallar üzerinden toplam; $\theta=\frac12$`atan2`$(2g_{xy}, g_{xx}-g_{yy})$; $F_\theta$ formülü. `F` ve `theta` `(B, H, W)`.
- `channel_sum_grad(img)` — kanalların Sobel büyüklüklerinin toplamı (karşılaştırma için).
- `mahalanobis_mask(img, samples, D0)` — `samples (N, 3)`; $a$ ortalama, $C$ kovaryans (`torch.cov`, yansız), $\sqrt{(z-a)^TC^{-1}(z-a)}\le D_0$ → `(B, H, W)` bool.
- `box_mask(img, samples, k=1.25)` — her kanalda $|z_j-a_j|\le k\sigma_j$ (yansız std).

Kontrol, $F_\theta^2$'nin $2\times2$ matrisin en büyük özdeğerine eşit olduğunu ve notlardaki örneği (aynı kanal gradyanları, farklı renk kenarı) sınar.
