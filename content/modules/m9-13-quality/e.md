`ms_ssim(x, y, R=255.0, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333))` — `x, y (B, H, W)` `float64` tensör → `(B,)`; tüm yığın **aynı anda**:

1. Her ölçekte d'deki Gauss süzgeci (simetrik yansıma), $l$ ve $cs$ haritaları, kenardan 5 kırpılmış ortalamalar: $\overline{cs}_j$ ve $\overline{l\cdot cs}_j$.
2. Ölçekler arası `F.avg_pool2d(…, 2)`.
3. $\text{MS-SSIM}=\prod_{j=1}^{M-1}\mathrm{relu}(\overline{cs}_j)^{w_j}\cdot\mathrm{relu}(\overline{l\cdot cs}_M)^{w_M}$.

Ölçek döngüsü (5 tekrar) serbest; görüntü döngüsü yok. En küçük boyut $11\cdot2^4=176$'dan büyük olmalı.
