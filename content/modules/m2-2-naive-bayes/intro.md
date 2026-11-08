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

:::türetim Naive Bayes karar kuralı ve Laplace düzeltmesi nereden geliyor?
**Bayes:** $P(y|x_1,\dots,x_d)=\frac{P(y)P(x_1,\dots,x_d|y)}{P(x)}$. Payda tüm sınıflar için aynı olduğundan sınıflandırma için $\arg\max_yP(y)P(x|y)$ yeter.

**"Naif" varsayım:** öznitelikler sınıf verildiğinde koşullu bağımsız: $P(x_1,\dots,x_d|y)=\prod_jP(x_j|y)$. Bunsuz $d$ ikili öznitelik için sınıf başına $2^d-1$ olasılık kestirmek gerekirdi; varsayımla yalnız $d$. Bedel: ilişkili öznitelikler kanıtı çift sayar ve olasılıklar aşırı uçlara itilir (sıralama/karar çoğu zaman yine iyidir, olasılık kalibrasyonu kötüdür).

**Sayısal kararlılık:** çarpım yerine log-toplam: $\log P(y)+\sum_j\log P(x_j|y)$ (çok sayıda küçük olasılığın çarpımı alttan taşar).

**Laplace düzeltmesi:** sayımdan en çok olabilirlik $\hat P(x_j=v|y)=\frac{n_{y,v}}{n_y}$; görülmemiş bir değer için 0 ve tüm çarpım 0 (başka hiçbir kanıt kurtaramaz). Her sayıma $\alpha$ eklemek, $V$ olası değer için $\frac{n_{y,v}+\alpha}{n_y+\alpha V}$ verir. Bayesçi yorum: Dirichlet($\alpha$) önseliyle posterior ortalaması (α = 1: düzgün önsel). Sıfır sorunu "veri yokken tam emin olma" hatasıdır; önsel bunu yumuşatır.
:::
