Bayes kuralı, sınıfın veriye göre olasılığını verinin sınıfa göre olasılığından çevirir:

$$
p(y\mid x)=\frac{p(y)\,p(x\mid y)}{p(x)}\;\propto\;p(y)\,p(x\mid y).
$$

$p(x\mid y)$'yi $d$ boyutlu $x$ için öğrenmek çok veri ister. **Naive** varsayım: sınıf bilindiğinde özellikler birbirinden bağımsız,

$$
p(y\mid x)\;\propto\;p(y)\prod_{j=1}^{d}p(x_j\mid y)
\qquad\Rightarrow\qquad
\log p(y\mid x)=\log p(y)+\sum_j\log p(x_j\mid y)+\text{sabit}.
$$

Her terim sayılarak (MLE, M0.4) tahmin edilir. Üç yaygın tür: **kategorik** (her özellik bir kategori), **Gaussian** (her özellik sınıfa göre bir normal dağılım) ve **multinomial** (kelime sayımları).

Bir değer bir sınıfta hiç görülmediyse MLE ona 0 verir ve çarpımın tamamı 0 olur. [[laplace düzeltmesi|Laplace düzeltmesi]] her sayıma $\alpha$ ekler: $\hat p=\frac{n_{v,c}+\alpha}{N_c+\alpha V}$.
