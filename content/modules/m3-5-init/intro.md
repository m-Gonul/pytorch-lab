Bir lineer katmanda $z=\sum_{j=1}^{n}w_jx_j$; terimler bağımsız ve sıfır ortalamalıysa

$$
\operatorname{Var}(z)=n\,\operatorname{Var}(w)\,\mathbb E[x^2].
$$

$L$ katman boyunca bu çarpan $L$ kez uygulanır: 1'den biraz büyükse aktivasyonlar patlar, biraz küçükse söner. Çözüm çarpanı 1 yapan ağırlık varyansı:

| başlatma | $\operatorname{Var}(w)$ | hangi aktivasyon |
|---|---|---|
| [[xavier init|Xavier/Glorot]] | $\frac{2}{n_{in}+n_{out}}$ | tanh, lineer |
| [[kaiming init|Kaiming/He]] | $\frac{2}{n_{in}}$ | ReLU ailesi |

ReLU sıfır ortalamalı simetrik bir girdinin ikinci momentini yarıya indirir: $\mathbb E[\text{ReLU}(z)^2]=\frac12\operatorname{Var}(z)$. Kaiming'in 2'si bunu telafi eder.

Geri geçişte de aynı hikâye; ek olarak aktivasyonun türevi çarpılır. Sigmoid'in türevi en fazla $0.25$; 10 katmanda gradyan en iyi ihtimalle $10^{-6}$ kat küçülür.
