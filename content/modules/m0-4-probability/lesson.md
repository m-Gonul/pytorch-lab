### Neden olasılık? {#neden}

Makine öğrenmesinde belirsizlik her yerdedir: ölçümler gürültülüdür, veri dünyanın küçük bir örneklemidir, aynı girdiye birden çok doğru cevap olabilir ("bu bulanık rakam 1 mi 7 mi?"). Olasılık kuramı bu belirsizliği hesaplanabilir hâle getirir. Bu modülün iki hedefi var:

1. Temel olasılık dilini (rastgele değişken, dağılım, beklenen değer, koşullu olasılık, Bayes) kesin biçimde kurmak.
2. **En çok olabilirlik (maximum likelihood, MLE)** ilkesini türetmek ve derin öğrenmedeki kayıp fonksiyonlarının (MSE, ikili çapraz entropi, çapraz entropi) aslında birer **negatif log-olabilirlik** olduğunu göstermek. Bir kaybı "neden bu kayıp?" diye sorabilmenin yolu budur.

**Model = verinin nasıl üretildiğine dair varsayım.** "Etiket, modelin tahmini etrafında Gauss gürültüsüyle dağılır" dediğinde MSE'yi, "etiket, modelin verdiği olasılıkla 1 olan bir yazı-turadır" dediğinde ikili çapraz entropiyi seçmiş olursun. Kayıp seçimi, veri hakkında bir iddiadır.

### Temel kavramlar: örnek uzayı, olay, olasılık {#temel}

:::tanım Olasılık uzayı (Kolmogorov aksiyomları)
- **Örnek uzayı** $\Omega$: bir deneyin olası tüm sonuçları (zar: $\{1,\dots,6\}$).
- **Olay** $A\subseteq\Omega$: sonuçlar kümesi ("çift gelir" $=\{2,4,6\}$).
- **Olasılık** $P$: olaylara sayı atar ve üç aksiyomu sağlar: (1) $P(A)\ge0$; (2) $P(\Omega)=1$; (3) ayrık (kesişmeyen) olaylar için $P(A_1\cup A_2\cup\dots)=P(A_1)+P(A_2)+\dots$.

Diğer her kural bunlardan türer: $P(\bar A)=1-P(A)$, $P(\varnothing)=0$, $P(A\cup B)=P(A)+P(B)-P(A\cap B)$.
:::

**Rastgele değişken** $X$, her sonuca bir sayı atayan bir fonksiyondur (iki zar atıldığında "toplam"). İki tür:

| | ayrık (discrete) | sürekli (continuous) |
|---|---|---|
| örnek | zar, sınıf etiketi, kelime | boy, piksel yoğunluğu, ağırlık |
| tanımlayan fonksiyon | **PMF** $p(x)=P(X=x)$, $\sum_xp(x)=1$ | **PDF** $p(x)\ge0$, $\int p(x)\,dx=1$ |
| bir aralığın olasılığı | $\sum_{x\in A}p(x)$ | $\int_ap(x)\,dx$ (eğri altındaki alan) |
| tek bir noktanın olasılığı | $p(x)$ | **0** |

:::yanılgı "Yoğunluk bir olasılıktır, 1'i aşamaz"
Sürekli bir dağılımda $p(x)$ olasılık değil, **birim uzunluk başına** olasılıktır (birimi $1/[x]$). $\mathcal N(0,\ 0.1^2)$'nin tepe yoğunluğu $\approx3.99$'dur. Sınırlı olan, yoğunluğun **alanıdır** (1). Bu yüzden sürekli modellerde log-olabilirlik pozitif olabilir ve "NLL negatif çıktı" bir hata değildir.
:::

**CDF** (birikimli dağılım fonksiyonu) $F(x)=P(X\le x)$ hem ayrık hem sürekli için tanımlıdır; sürekli durumda $p(x)=F'(x)$.

### Ortak, marjinal ve koşullu olasılık {#kosullu}

İki rastgele değişken birlikte: **ortak dağılım** $p(x,y)$. Ondan:

- **Marjinal** (toplam kuralı): $p(x)=\sum_yp(x,y)$ (süreklide $\int$). $y$'yi "toplayıp atarız".
- **Koşullu**: $p(y\mid x)=\dfrac{p(x,y)}{p(x)}$, $p(x)>0$ iken. $X=x$ olduğunu bildiğimiz dünyada $Y$'nin dağılımı.
- **Çarpım kuralı**: $p(x,y)=p(y\mid x)\,p(x)=p(x\mid y)\,p(y)$. Çok değişkene zincirlenir: $p(x_1,\dots,x_n)=\prod_ip(x_i\mid x_1,\dots,x_{i-1})$ — dil modellerinin **otoregresif** çarpanlarına ayırması tam olarak budur (M4.6).
- **Bağımsızlık**: $X\perp Y\iff p(x,y)=p(x)p(y)$ her $x,y$ için $\iff p(y\mid x)=p(y)$. **Koşullu bağımsızlık** $X\perp Y\mid Z$: $p(x,y\mid z)=p(x\mid z)p(y\mid z)$.

:::örnek Tablo üzerinden
100 e-posta: 30 spam. Spam olanların 20'sinde "ücretsiz" kelimesi geçiyor, spam olmayanların 7'sinde. $P(\text{ücretsiz}\mid\text{spam})=20/30\approx0.67$, $P(\text{ücretsiz})=27/100$, $P(\text{spam}\mid\text{ücretsiz})=20/27\approx0.74$. Koşulun yönü değişince sayı da değişir: $P(A\mid B)\ne P(B\mid A)$.
:::

### Bayes kuralı {#bayes}

Çarpım kuralının iki biçimini eşitleyince:
$$\underbrace{p(h\mid d)}_{\text{sonsal}}=\frac{\overbrace{p(d\mid h)}^{\text{olabilirlik}}\;\overbrace{p(h)}^{\text{önsel}}}{\underbrace{p(d)}_{\text{kanıt}}},\qquad p(d)=\sum_{h'}p(d\mid h')p(h').$$
$h$ hipotez (hasta mı, hangi sınıf, hangi parametre), $d$ veri. Bayes kuralı "verinin hipotez altında ne kadar olası olduğunu" (çoğu zaman bildiğimiz şey) "veriyi gördükten sonra hipotezin olasılığına" (istediğimiz şey) çevirir.

:::örnek Tıbbi test
Hastalık nüfusun %1'inde var. Test hastaların %90'ını yakalar (duyarlılık), sağlıklıların %95'ini negatif bulur (özgüllük). Pozitif çıktın: hasta olma olasılığın?
$$P(H\mid+)=\frac{0.9\cdot0.01}{0.9\cdot0.01+0.05\cdot0.99}=\frac{0.009}{0.0585}\approx0.154.$$
Yalnız **%15**. Çoğu insanın sezgisi %90 der: önseli (taban oranı) unutmak. 1000 kişide: 10 hasta → 9 pozitif; 990 sağlıklı → ~50 pozitif. Pozitiflerin yalnız 9/59'u hasta.
:::

Aşağıda ızgarayla dene: yaygınlığı değiştir, testi iyileştir, ikinci bir testi aç (ilk testin sonsalı ikincinin önseli olur).

::lab l1

**Makine öğrenmesindeki karşılığı:** bir sınıflandırıcının çıktısı $p(y\mid x)$ bir sonsaldır ve eğitim verisindeki sınıf oranlarını (önsel) içinde taşır. Test ortamında sınıf oranları değişirse (eğitimde %50 spam, gerçekte %5) olasılıklar yanlış kalibre olur; Bayes kuralıyla önsel düzeltmesi yapılabilir: $p_{\text{yeni}}(y\mid x)\propto p(y\mid x)\,\frac{\pi_{\text{yeni}}(y)}{\pi_{\text{eğitim}}(y)}$.

### Beklenen değer, varyans, kovaryans {#beklenen}

:::tanım Beklenen değer ve varyans
$$\mathbb E[X]=\sum_xx\,p(x)\quad\Big(\text{süreklide }\int x\,p(x)\,dx\Big),\qquad \mathbb E[g(X)]=\sum_xg(x)p(x).$$
$$\mathrm{Var}[X]=\mathbb E\big[(X-\mathbb E X)^2\big]=\mathbb E[X^2]-(\mathbb EX)^2,\qquad \sigma=\sqrt{\mathrm{Var}[X]}\ (\text{X ile aynı birimde}).$$
Beklenen değer bir "ağırlık merkezi", varyans "ortalamadan karesel uzaklığın ortalaması"dır.
:::

**Kurallar** (kanıtları tanımdan bir satırdır):

| kural | koşul |
|---|---|
| $\mathbb E[aX+bY+c]=a\,\mathbb EX+b\,\mathbb EY+c$ | **her zaman** (bağımsızlık gerekmez) |
| $\mathrm{Var}[aX+c]=a^2\mathrm{Var}[X]$ | her zaman (kaydırma varyansı değiştirmez) |
| $\mathrm{Var}[X+Y]=\mathrm{Var}X+\mathrm{Var}Y+2\,\mathrm{Cov}(X,Y)$ | her zaman |
| $\mathrm{Var}[X+Y]=\mathrm{Var}X+\mathrm{Var}Y$ | $X,Y$ ilişkisizse ($\mathrm{Cov}=0$) |
| $\mathbb E[XY]=\mathbb EX\,\mathbb EY$ | ilişkisizse (bağımsızlık yeter) |

$\mathrm{Cov}(X,Y)=\mathbb E[(X-\mathbb EX)(Y-\mathbb EY)]$; korelasyon $\rho=\mathrm{Cov}/(\sigma_X\sigma_Y)\in[-1,1]$ (Cauchy–Schwarz, M0.1). **İlişkisiz ≠ bağımsız:** $X\sim U(-1,1)$, $Y=X^2$ tamamen $X$'e bağlıdır ama $\mathrm{Cov}=\mathbb E[X^3]=0$.

:::türetim Ortalamanın standart hatası: σ/√n
$X_1,\dots,X_n$ bağımsız, her biri ortalama $\mu$, varyans $\sigma^2$. $\bar X=\frac1n\sum X_i$.
$\mathbb E\bar X=\frac1n\sum\mathbb EX_i=\mu$ (doğrusallık). $\mathrm{Var}\bar X=\frac1{n^2}\mathrm{Var}\sum X_i=\frac1{n^2}\cdot n\sigma^2=\frac{\sigma^2}n$ (bağımsızlıkla kovaryans terimleri 0). Dolayısıyla $\mathrm{sd}(\bar X)=\sigma/\sqrt n$.
**Sonuçları:** hatayı yarıya indirmek için 4 kat veri gerekir. Mini-batch gradyanı $B$ örneğin ortalamasıdır: gürültüsü $1/\sqrt B$ ile azalır, batch'i 4 katına çıkarmak gürültüyü yarıya indirir (M1.2). **Koşul:** bağımsızlık. Örnekler ilişkiliyse (ardışık video kareleri) etkin örnek sayısı $n$'den küçüktür.
:::

**Çok değişkenli:** rastgele vektör $x\in\mathbb R^d$ için ortalama vektörü $\mu=\mathbb E[x]$ ve **kovaryans matrisi** $\Sigma=\mathbb E[(x-\mu)(x-\mu)^\top]$ ($d\times d$, simetrik, pozitif yarı-tanımlı). $\Sigma_{ii}$ varyanslar, $\Sigma_{ij}$ kovaryanslar. Doğrusal dönüşüm: $\mathrm{Cov}(Ax)=A\Sigma A^\top$ — ağırlık başlatmada aktivasyon varyansının katmandan katmana nasıl değiştiğinin hesabı (M3.5) bu formüldür.

### Temel dağılımlar {#dagilimlar}

| dağılım | değer | PMF / PDF | ortalama | varyans | ML'de nerede |
|---|---|---|---|---|---|
| Bernoulli($p$) | $\{0,1\}$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | ikili sınıflandırma, dropout maskesi |
| Binom($m,p$) | $0..m$ | $\binom mk p^k(1-p)^{m-k}$ | $mp$ | $mp(1-p)$ | $m$ Bernoulli'nin toplamı |
| Kategorik($\pi$) | $1..K$ | $\pi_k$ | — | — | çok sınıflı sınıflandırma, dil modeli |
| Gauss $\mathcal N(\mu,\sigma^2)$ | $\mathbb R$ | $\frac1{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | regresyon gürültüsü, başlatma, VAE, difüzyon |
| Laplace($\mu,b$) | $\mathbb R$ | $\frac1{2b}e^{-|x-\mu|/b}$ | $\mu$ | $2b^2$ | L1 kaybı (MAE), seyreklik önseli |
| Üstel($\lambda$) | $[0,\infty)$ | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | bekleme süreleri |
| Düzgün $U(a,b)$ | $[a,b]$ | $\frac1{b-a}$ | $\frac{a+b}2$ | $\frac{(b-a)^2}{12}$ | rastgele başlatma, veri artırma |

**Çok değişkenli Gauss** $\mathcal N(\mu,\Sigma)$: $p(x)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\!\big(-\tfrac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\big)$. Eş-yoğunluk eğrileri elipslerdir; eksenleri $\Sigma$'nın özvektörleri, yarı-eksen uzunlukları özdeğerlerin kareköküdür. $\Sigma=\sigma^2I$ ise çember (izotropik).

<figure class="fig wide"><svg viewBox="0 0 640 190" role="img" aria-label="Üç dağılım: Bernoulli çubukları, Gauss çanı ve üstel azalan eğri"><g transform="translate(20,10)"><line x1="0" y1="150" x2="170" y2="150" class="ink"/><rect x="35" y="45" width="34" height="105" class="soft-a"/><rect x="35" y="45" width="34" height="105" class="ca"/><rect x="105" y="105" width="34" height="45" class="soft-b"/><rect x="105" y="105" width="34" height="45" class="cb"/><text x="52" y="168" text-anchor="middle" class="sm">0</text><text x="122" y="168" text-anchor="middle" class="sm">1</text><text x="52" y="38" text-anchor="middle" class="sm">0.7</text><text x="122" y="98" text-anchor="middle" class="sm">0.3</text><text x="85" y="12" text-anchor="middle" class="sm">Bernoulli(0.3): çubuklar = olasılık</text></g><g transform="translate(230,10)"><line x1="0" y1="150" x2="180" y2="150" class="ink"/><path d="M0,148.9 L5,148.3 L10,147.3 L15,145.8 L20,143.8 L25,140.9 L30,137.1 L35,132.3 L40,126.3 L45,119.2 L50,110.9 L55,101.9 L60,92.4 L65,82.9 L70,73.9 L75,66.2 L80,60.1 L85,56.3 L90,55.0 L95,56.3 L100,60.1 L105,66.2 L110,73.9 L115,82.9 L120,92.4 L125,101.9 L130,110.9 L135,119.2 L140,126.3 L145,132.3 L150,137.1 L155,140.9 L160,143.8 L165,145.8 L170,147.3 L175,148.3 L180,148.9 Z" class="soft-p"/><path d="M0,148.9 L5,148.3 L10,147.3 L15,145.8 L20,143.8 L25,140.9 L30,137.1 L35,132.3 L40,126.3 L45,119.2 L50,110.9 L55,101.9 L60,92.4 L65,82.9 L70,73.9 L75,66.2 L80,60.1 L85,56.3 L90,55.0 L95,56.3 L100,60.1 L105,66.2 L110,73.9 L115,82.9 L120,92.4 L125,101.9 L130,110.9 L135,119.2 L140,126.3 L145,132.3 L150,137.1 L155,140.9 L160,143.8 L165,145.8 L170,147.3 L175,148.3 L180,148.9" class="cp w2" fill="none"/><line x1="90" y1="150" x2="90" y2="55" class="mute dash"/><text x="90" y="168" text-anchor="middle" class="sm">μ</text><line x1="60" y1="92.4" x2="120" y2="92.4" class="ink"/><line x1="60" y1="150" x2="60" y2="92.4" class="mute dash"/><line x1="120" y1="150" x2="120" y2="92.4" class="mute dash"/><text x="90" y="87.4" text-anchor="middle" class="sm">±σ: %68</text><text x="90" y="12" text-anchor="middle" class="sm">Gauss: alan = olasılık</text></g><g transform="translate(450,10)"><line x1="0" y1="150" x2="175" y2="150" class="ink"/><path d="M5,40 C25,85 45,115 70,130 C95,142 130,147 175,149" class="cq w2" fill="none"/><text x="5" y="168" class="sm">0</text><text x="88" y="12" text-anchor="middle" class="sm">Üstel: hep azalan</text></g></svg><figcaption>Şekil 1 — Ayrık dağılımda çubuk yüksekliği olasılıktır; sürekli dağılımda olasılık eğri altındaki alandır. Gauss'ta μ ± σ aralığı olasılığın ~%68'ini, ±2σ ~%95'ini taşır.</figcaption></figure>

Aşağıdaki deneyde bir dağılımdan örnek çek: histogram teorik eğriye yaklaşır, örneklem ortalaması gerçek ortalamaya — ve bu yaklaşmanın hızı $\sigma/\sqrt n$'dir.

::lab l2

### Büyük sayılar ve merkezi limit {#mlt}

**Büyük sayılar yasası (BSY):** bağımsız, özdeş dağılmış (i.i.d.) $X_i$'lerin ortalaması $n\to\infty$ iken $\mu$'ye yakınsar (sonlu ortalama yeter). Eğitimde "kaybın veri üzerindeki ortalaması, gerçek beklenen kaybın tahminidir" demenin dayanağı budur: **ampirik risk** $\frac1n\sum\ell(x_i)$, **gerçek risk** $\mathbb E[\ell(X)]$'in tahminidir.

**Merkezi limit teoremi (MLT):** sonlu varyanslı i.i.d. $X_i$ için
$$\frac{\bar X_n-\mu}{\sigma/\sqrt n}\ \xrightarrow{\ d\ }\ \mathcal N(0,1).$$
Temel dağılım çarpık, ayrık, iki tepeli olabilir; ortalamanın dağılımı yine Gauss'a yaklaşır. **Ne demez:** tek tek $X_i$'lerin Gauss olduğunu; küçük $n$'de yaklaşımın iyi olduğunu (çarpık dağılımlarda yavaştır); varyans sonsuzsa da geçerli olduğunu (Cauchy karşı-örneği aşağıda).

Gauss'un her yerde görünmesinin sebebi MLT'dir: birçok küçük bağımsız etkinin toplamı (ölçüm hatası, mini-batch gradyanı) yaklaşık Gauss'tur.

::lab l3

### Olabilirlik: aynı formül, farklı değişken {#olabilirlik}

Bir parametrik model $p(x\mid\theta)$ verelim. $\theta$ sabitken $x$'in fonksiyonu olarak bu bir **olasılık** (dağılım). $x$ (gözlenen veri) sabitken **$\theta$'nın fonksiyonu** olarak düşünürsek bu **olabilirlik**tir (likelihood):
$$L(\theta)=p(\mathcal D\mid\theta)\overset{\text{i.i.d.}}{=}\prod_{i=1}^np(x_i\mid\theta).$$
Olabilirlik $\theta$ üzerinde bir dağılım **değildir**: $\int L(\theta)\,d\theta$ genelde 1 değildir. "Bu parametre değeri, gördüğüm veriyi ne kadar olası kılıyor?" sorusunun cevabıdır.

**Neden log?** Üç nedenle her zaman $\ell(\theta)=\log L(\theta)=\sum_i\log p(x_i\mid\theta)$ ile çalışırız:

1. **Sayısal:** 1000 örnek, her biri olasılık ~0.1 → $L\approx10^{-1000}$; float64'ün en küçük pozitif sayısı ~$10^{-308}$. Çarpım **0'a taşar** (underflow); log-toplam $\approx-2302.6$ rahatça temsil edilir.
2. **Matematiksel:** çarpımın türevi karmaşık, toplamın türevi terimlerin türevlerinin toplamıdır. Ayrıca toplam, mini-batch ile tahmin edilebilir (BSY).
3. **Aynı en iyi nokta:** $\log$ kesin artan olduğu için $\arg\max L=\arg\max\ell$.

### En çok olabilirlik kestirimi (MLE) {#mle}

$$\hat\theta_{\text{MLE}}=\arg\max_\theta\ \sum_{i=1}^n\log p(x_i\mid\theta)=\arg\min_\theta\ \underbrace{-\frac1n\sum_{i=1}^n\log p(x_i\mid\theta)}_{\text{NLL}}.$$
Optimizasyon kütüphaneleri küçülttüğü için negatif log-olabilirlik (NLL) küçültülür; $1/n$ ile ortalamak en iyi noktayı değiştirmez ama ölçeği veri boyutundan bağımsız yapar.

:::türetim Bernoulli: p̂ = k/n
$n$ atışta $k$ tura: $\ell(p)=k\log p+(n-k)\log(1-p)$. Türev: $\ell'(p)=\frac kp-\frac{n-k}{1-p}$. Sıfıra eşitle: $k(1-p)=(n-k)p\Rightarrow p=k/n$. İkinci türev $-\frac k{p^2}-\frac{n-k}{(1-p)^2}<0$: maksimum. **Koşul:** $0<k<n$; $k=0$ ya da $n$ ise maksimum sınırda ($p=0$ ya da 1) ve türev sıfır olmaz — tahmin yine $k/n$ ama aşırı güvenlidir (aşağıda MAP).
:::

:::türetim Gauss: μ̂ = x̄, σ̂² = (1/n)Σ(xᵢ − x̄)²
$\ell(\mu,\sigma^2)=-\frac n2\log(2\pi\sigma^2)-\frac1{2\sigma^2}\sum_i(x_i-\mu)^2$.
$\partial\ell/\partial\mu=\frac1{\sigma^2}\sum_i(x_i-\mu)=0\Rightarrow\hat\mu=\bar x$ — $\sigma$'dan bağımsız.
$\partial\ell/\partial\sigma^2=-\frac n{2\sigma^2}+\frac1{2\sigma^4}\sum_i(x_i-\mu)^2=0\Rightarrow\hat\sigma^2=\frac1n\sum_i(x_i-\hat\mu)^2$.
:::

:::türetim Kategorik: p̂ₖ = nₖ/n (Lagrange çarpanı)
$\ell(\pi)=\sum_kn_k\log\pi_k$, kısıt $\sum_k\pi_k=1$. Lagrange: $\mathcal L=\sum_kn_k\log\pi_k+\lambda(1-\sum_k\pi_k)$. $\partial/\partial\pi_k$: $n_k/\pi_k-\lambda=0\Rightarrow\pi_k=n_k/\lambda$; kısıttan $\lambda=n$. Yani frekanslar. Kısıtsız maksimize etmeye çalışsaydık her $\pi_k\to\infty$ olurdu: kısıt (olasılıkların toplamı 1) şarttır. Sinir ağlarında bu kısıtı **softmax** kendiliğinden sağlar.
:::

Aşağıda yazı-tura olabilirliğini çiz: $n$ ve $k$'yı değiştir, eğrinin tepesinin $k/n$'de olduğunu ve $n$ büyüdükçe daraldığını gör. Önsel gücünü artırarak MAP'e geç.

::lab l4

### MLE'nin özellikleri ve sınırları {#ozellikler}

- **Tutarlılık:** model doğruysa (veri gerçekten $p(x\mid\theta^*)$'dan geliyorsa) ve bazı düzenlilik koşulları altında $\hat\theta_n\to\theta^*$.
- **Asimptotik normallik:** büyük $n$'de $\hat\theta$ yaklaşık Gauss dağılır, varyansı $\frac1{nI(\theta)}$ ($I$: Fisher bilgisi; log-olabilirliğin ortalama eğriliği). Bernoulli'de $\mathrm{Var}\hat p\approx p(1-p)/n$. Eğri ne kadar sivriyse tahmin o kadar kesindir — deneyde gördüğün daralma.
- **Yanlılık olabilir:** Gauss varyansının MLE'si sistematik olarak küçüktür.

:::türetim MLE varyansı neden yanlı? E[σ̂²] = (n−1)/n · σ²
$\sum_i(x_i-\bar x)^2=\sum_i(x_i-\mu)^2-n(\bar x-\mu)^2$ (karesini açıp $\sum_i(x_i-\mu)=n(\bar x-\mu)$ kullan). Beklenen değer: $n\sigma^2-n\cdot\frac{\sigma^2}n=(n-1)\sigma^2$. Böylece $\mathbb E[\hat\sigma^2_{\text{MLE}}]=\frac{n-1}n\sigma^2<\sigma^2$.
**Sezgi:** $\bar x$ verinin kendisinden hesaplandığı için verilere gerçek $\mu$'den daha yakındır; bir "serbestlik derecesi" harcandı. Yansız kestirici $\frac1{n-1}\sum(x_i-\bar x)^2$ (Bessel düzeltmesi). PyTorch: `x.var()` varsayılan olarak yansız ($n-1$; `correction=1`), NumPy `x.var()` varsayılan olarak MLE ($n$; `ddof=0`). BatchNorm normalize ederken $n$, koşan istatistikleri güncellerken $n-1$ kullanır.
:::

- **Az veride aşırı güven:** 3 atışta 3 tura → $\hat p=1$: "yazı gelmesi imkânsız". Bir dil modeli eğitim verisinde hiç görmediği kelimeye 0 olasılık verirse, test cümlesinde o kelime çıktığında log-olabilirlik $-\infty$ olur.

### MAP: önsel eklemek = düzenlileştirme {#map}

Bayes kuralını parametrelere uygula: $p(\theta\mid\mathcal D)\propto p(\mathcal D\mid\theta)\,p(\theta)$. **MAP** kestirimi sonsalın tepesidir:
$$\hat\theta_{\text{MAP}}=\arg\max_\theta\Big[\sum_i\log p(x_i\mid\theta)+\log p(\theta)\Big].$$
İkinci terim bir **düzenlileştirici**dir:

| önsel $p(\theta)$ | $-\log p(\theta)$ | adı |
|---|---|---|
| $\mathcal N(0,\tau^2I)$ | $\frac1{2\tau^2}\|\theta\|_2^2+c$ | L2 / ağırlık sönümü (weight decay), M1.3 |
| Laplace$(0,b)$ | $\frac1b\|\theta\|_1+c$ | L1 / seyreklik, lasso |
| Beta$(a,a)$, Bernoulli için | $-(a-1)[\log p+\log(1-p)]$ | Laplace düzeltmesi: $\hat p=\frac{k+a-1}{n+2a-2}$ |

$a=2$ ile Beta önseli "her iki sonuçtan birer sahte gözlem ekle" demektir: 3/3 tura → $\hat p=4/5$. Veri arttıkça önselin etkisi söner ($n\to\infty$'da MAP → MLE).

**Tam Bayesçi yaklaşım** tek bir nokta yerine sonsalın tamamını kullanır (tahminlerde $\theta$ üzerinden ortalama alır); belirsizliği taşır ama derin ağlarda hesaplaması zordur (yaklaşık yöntemler: MC dropout, ansambller, varyasyonel çıkarım — M5.1).

### Kayıp fonksiyonları birer NLL'dir {#kayiplar}

Model bir girdi $x$ için çıktı dağılımının parametrelerini üretsin. NLL'yi yaz, sabitleri at:

:::türetim Gauss gürültüsü → MSE
Varsayım: $y\mid x\sim\mathcal N(f_\theta(x),\sigma^2)$, $\sigma$ sabit. $-\log p(y\mid x)=\frac{(y-f_\theta(x))^2}{2\sigma^2}+\frac12\log(2\pi\sigma^2)$. İkinci terim ve $\frac1{2\sigma^2}$ çarpanı $\theta$'ya bağlı değil: NLL'yi küçültmek $\sum(y-f_\theta(x))^2$'yi, yani **MSE**'yi küçültmektir. **Varsayımın sonucu:** MSE aykırı değerlere çok duyarlıdır (Gauss kuyrukları incedir); Laplace gürültüsü varsayarsan **MAE** ($|y-f|$) çıkar. $\sigma$'yı da model tahmin ederse (heteroskedastik regresyon) kayıp $\frac{(y-\mu)^2}{2\sigma^2}+\log\sigma$ olur: model belirsiz olduğu yerde $\sigma$'yı büyütebilir.
:::

:::türetim Bernoulli → ikili çapraz entropi
$y\in\{0,1\}$, $p=\sigma(z)$ modelin olasılığı. $-\log p(y\mid x)=-[y\log p+(1-y)\log(1-p)]$ — **BCE**. Pratikte $z$ (logit) üzerinden yazılır (`BCEWithLogitsLoss`): $\log\sigma(z)=-\mathrm{softplus}(-z)$ kararlı biçimde hesaplanır, $\sigma$ önce hesaplanıp sonra log alınırsa büyük $|z|$'de $\log0$ olur.
:::

:::türetim Kategorik → çapraz entropi
$y\in\{1..K\}$, $\pi=\mathrm{softmax}(z)$. $-\log p(y\mid x)=-\log\pi_y=-z_y+\log\sum_ke^{z_k}$ — **çapraz entropi** (M1.5). İkinci terim **logsumexp**'tir; kararlı hesabı için en büyük logit çıkarılır: $\log\sum e^{z_k}=m+\log\sum e^{z_k-m}$.
:::

**Bilgi kuramı bağlantısı.** Gerçek veri dağılımı $p_{\text{data}}$, model $q_\theta$. Beklenen NLL:
$$\mathbb E_{x\sim p_{\text{data}}}[-\log q_\theta(x)]=\underbrace{H(p_{\text{data}})}_{\text{entropi, sabit}}+\underbrace{\mathrm{KL}(p_{\text{data}}\,\|\,q_\theta)}_{\ge0}.$$
Sol yan **çapraz entropi** $H(p,q)$'dir. Entropi modele bağlı olmadığı için **NLL'yi küçültmek = KL ıraksamasını küçültmek** = modeli veri dağılımına yaklaştırmak. KL'nin her zaman $\ge0$ olması (Gibbs eşitsizliği, Jensen'den) ve yalnız $q=p$ iken 0 olması, MLE'nin "doğru" hedef olmasının bilgi kuramsal gerekçesidir. **Dikkat:** KL simetrik değildir; $\mathrm{KL}(p\|q)$ küçültmek $q$'yu $p$'nin tüm tepelerini kapsamaya zorlar ("kütle kapsayan"), ters yön ise tek tepeye odaklanır (VAE ve varyasyonel çıkarımda önemli, M5.1).

### Bootstrap: tahminin belirsizliği {#bootstrap}

Bir istatistik hesapladık (ortalama, medyan, doğruluk). Ne kadar güvenmeliyiz? Gerçek dağılımdan yeni örneklemler çekemiyoruz; **bootstrap** elimizdeki veriyi dağılımın yerine koyar:

1. Veriden **yerine koyarak** $n$ örnek çek (bazı örnekler tekrar eder, ~%37'si hiç seçilmez).
2. İstatistiği hesapla.
3. $B$ kez (ör. 1000) tekrarla. $B$ değerin standart sapması **standart hata** tahminidir; %2.5 ve %97.5 yüzdelikleri yaklaşık %95 **güven aralığı**dır (yüzdelik yöntemi).

<figure class="fig"><svg viewBox="0 0 420 170" role="img" aria-label="Bootstrap: bir veri kümesinden yerine koyarak çekilen çok sayıda yeniden örneklem ve istatistiklerinin histogramı"><g class="sm"><rect x="10" y="20" width="110" height="30" class="soft-a"/><rect x="10" y="20" width="110" height="30" class="ca"/><text x="65" y="40" text-anchor="middle">veri (n örnek)</text><line x1="120" y1="35" x2="160" y2="20" class="ink"/><line x1="120" y1="35" x2="160" y2="60" class="ink"/><line x1="120" y1="35" x2="160" y2="100" class="ink"/><rect x="160" y="8" width="90" height="22" class="mute"/><text x="205" y="23" text-anchor="middle">yeniden örn. 1 → θ̂₁</text><rect x="160" y="48" width="90" height="22" class="mute"/><text x="205" y="63" text-anchor="middle">yeniden örn. 2 → θ̂₂</text><text x="205" y="88" text-anchor="middle">⋮</text><rect x="160" y="90" width="90" height="22" class="mute"/><text x="205" y="105" text-anchor="middle">yeniden örn. B → θ̂_B</text><line x1="250" y1="60" x2="280" y2="60" class="ink"/><g transform="translate(285,30)"><rect x="0" y="80" width="12" height="10" class="soft-p"/><rect x="13" y="60" width="12" height="30" class="soft-p"/><rect x="26" y="35" width="12" height="55" class="soft-p"/><rect x="39" y="20" width="12" height="70" class="soft-p"/><rect x="52" y="30" width="12" height="60" class="soft-p"/><rect x="65" y="55" width="12" height="35" class="soft-p"/><rect x="78" y="75" width="12" height="15" class="soft-p"/><line x1="0" y1="90" x2="95" y2="90" class="ink"/><line x1="10" y1="0" x2="10" y2="95" class="cb dash"/><line x1="85" y1="0" x2="85" y2="95" class="cb dash"/><text x="48" y="110" text-anchor="middle">θ̂'ların dağılımı</text><text x="48" y="125" text-anchor="middle" class="tb">%2.5 – %97.5: güven aralığı</text></g></g></svg><figcaption>Şekil 2 — Bootstrap, "başka bir veri kümesi toplasaydık tahmin ne olurdu?" sorusunu elimizdeki verinin yeniden örneklemleriyle taklit eder.</figcaption></figure>

**Ne zaman yanılır:** veri gerçek dağılımı iyi temsil etmiyorsa (çok az veri, ağır kuyruk), istatistik çok "uç" bir şeye bağlıysa (maksimum), örnekler bağımsız değilse (zaman serisi: blok bootstrap gerekir). Model karşılaştırmada iki modelin doğruluk farkının bootstrap aralığı 0'ı içeriyorsa fark güvenilir değildir. Modülün **e** aşamasında bootstrap'ı döngüsüz, tek tensör işlemiyle (`torch.randint(n, (B, n))`) yapacaksın.

### PyTorch'ta olasılık {#pytorch}

```python
import torch
from torch.distributions import Normal, Bernoulli, Categorical
d = Normal(loc=torch.tensor(0.), scale=torch.tensor(2.))
x = d.sample((5,))            # (5,) örnek — gradyan akmaz
lp = d.log_prob(x)            # log yoğunluk, autograd ile türevlenebilir
Categorical(logits=torch.tensor([2., 0., -1.])).log_prob(torch.tensor(0))  # = -CE
```

- **Logit ile çalış:** `Bernoulli(logits=z)`, `Categorical(logits=z)`, `F.cross_entropy(z, y)`: olasılıkları açıkça hesaplayıp log almak sayısal olarak kırılgandır.
- **Kısıtlı parametreler:** $\sigma>0$'ı gradyan inişiyle korumak yerine **yeniden parametrelendir**: $\sigma=e^s$ ya da $\mathrm{softplus}(s)$, $s$ serbest (modülün **d** aşaması).
- **`sample` ile `rsample`:** `rsample` yeniden parametrelendirme hilesini kullanır ($x=\mu+\sigma\varepsilon$, $\varepsilon\sim\mathcal N(0,1)$): örnekten parametrelere gradyan akar (VAE, M5.1). `sample` gradyan taşımaz.
- **Tohum:** `torch.manual_seed` ile örnekleme tekrarlanabilir olur; GPU'da bazı işlemler yine de deterministik değildir (M0.2 sayısal konular).

### Özet ve sembol tablosu {#ozet}

| sembol | anlamı |
|---|---|
| $p(x)$ | PMF (ayrık) ya da PDF (sürekli, yoğunluk) |
| $p(x,y)$, $p(y\mid x)$ | ortak, koşullu dağılım |
| $\mathbb E[X]$, $\mathrm{Var}[X]$, $\mathrm{Cov}$ | beklenen değer, varyans, kovaryans |
| $L(\theta)$, $\ell(\theta)$ | olabilirlik, log-olabilirlik ($\theta$'nın fonksiyonu) |
| $\hat\theta_{\text{MLE}}$, $\hat\theta_{\text{MAP}}$ | en çok olabilirlik ve en büyük sonsal kestirimi |
| $H(p)$, $H(p,q)$, $\mathrm{KL}(p\|q)$ | entropi, çapraz entropi, KL ıraksaması |

**Akılda kalsın:** kayıp seçmek veri hakkında bir varsayım seçmektir (Gauss → MSE, Bernoulli → BCE, kategorik → CE). MLE log-olabilirliği büyütür, bu da modeli KL anlamında veri dağılımına yaklaştırır. Önsel eklemek düzenlileştirmedir. Ortalamaların hatası $\sigma/\sqrt n$ ile küçülür ve dağılımları Gauss'a yaklaşır — mini-batch gradyanının davranışını belirleyen iki gerçek.

**Kaynaklar:** Deisenroth, Faisal ve Ong, *Mathematics for Machine Learning*, Bölüm 6 (olasılık) ve 8.3 (parametre kestirimi); Murphy, *Probabilistic Machine Learning: An Introduction*, Bölüm 2–4; Blitzstein ve Hwang, *Introduction to Probability* (Bayes, beklenen değer, MLT); Goodfellow ve ark., *Deep Learning* Bölüm 3 ve 5.5; d2l.ai §2.6; Efron ve Tibshirani, *An Introduction to the Bootstrap*; PyTorch `torch.distributions` belgeleri.
