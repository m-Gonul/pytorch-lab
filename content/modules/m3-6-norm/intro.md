Normalizasyon katmanları aktivasyonları sıfır ortalama, birim varyansa çeker, sonra öğrenilen bir ölçek $\gamma$ ve kaydırma $\beta$ uygular:

$$
\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y=\gamma\odot\hat x+\beta .
$$

Hepsinin farkı $\mu$ ve $\sigma^2$'nin **hangi eksenler** üzerinden hesaplandığı ($N$ batch, $C$ kanal/özellik, $H\times W$ uzamsal):

| katman | istatistik eksenleri | $\gamma,\beta$ boyutu |
|---|---|---|
| BatchNorm | $N\,(,H,W)$ | $C$ |
| LayerNorm | $C\,(,H,W)$ | $C$ (ya da $C,H,W$) |
| InstanceNorm | $H,W$ | $C$ |
| GroupNorm | $C/G$ grubu $,H,W$ | $C$ |
| RMSNorm | $C$; ortalama çıkarılmaz, $\sqrt{\text{mean}(x^2)}$'e bölünür | $C$ |

BatchNorm eğitimde batch istatistiğini, değerlendirmede eğitim boyunca biriktirilen **running** istatistikleri kullanır. Geri geçişi, $\mu$ ve $\sigma$'nın tüm batch'e bağlı olması yüzünden üç terimlidir:

$$
dx=\frac{1}{N\sigma}\Big(N\,d\hat x-\sum_j d\hat x_j-\hat x\sum_j d\hat x_j\hat x_j\Big),\quad d\hat x=\gamma\,dy,\quad d\gamma=\sum dy\,\hat x,\quad d\beta=\sum dy .
$$
