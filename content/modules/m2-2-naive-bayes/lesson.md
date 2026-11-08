### Üretken ve ayırt edici modeller {#uretken}

Bir sınıflandırıcı sonunda $p(y\mid x)$'e karar verir. Oraya iki yoldan gidilebilir:

- **Ayırt edici (discriminative):** $p(y\mid x)$'i doğrudan modelle. Lojistik regresyon (M1.4), softmax (M1.5), sinir ağları. "Sınıfları ayıran sınır nerede?"
- **Üretken (generative):** her sınıfın verisinin **nasıl üretildiğini** modelle, $p(x\mid y)$, ve sınıf önselini $p(y)$; sonra Bayes kuralıyla çevir:
$$p(y\mid x)=\frac{p(x\mid y)\,p(y)}{\sum_{y'}p(x\mid y')\,p(y')}.$$
"Her sınıf neye benziyor?"

Üretken modeller daha fazla şey öğrenir (girdinin dağılımını), bu yüzden daha güçlü varsayımlar ister ama az veriyle iyi çalışabilir, eksik öznitelikleri doğal olarak işler ve yeni örnek üretebilir. **Naive Bayes** en basit üretken sınıflandırıcıdır; Gauss diskriminant analizi (LDA/QDA), gizli Markov modelleri ve — çok daha karmaşık biçimde — VAE ve difüzyon modelleri (Faz 5) aynı ailedendir.

### Bayes sınıflandırıcısı {#bayes-siniflandirici}

Gerçek $p(y\mid x)$'i bilseydik en iyi karar ne olurdu? Hata olasılığını en aza indirmek için her $x$'te **en olası sınıfı** seçmek:
$$\hat y(x)=\arg\max_y\ p(y\mid x)=\arg\max_y\ p(x\mid y)\,p(y).$$
(Payda $p(x)$ tüm sınıflarda aynıdır.) Bu kuralın hatası **Bayes hatası**dır: hiçbir sınıflandırıcı ondan iyi olamaz (M2.1'deki k-NN kuramının referans noktası). Yanlış sınıflandırma maliyetleri eşit değilse karar, beklenen maliyeti en aza indiren sınıf olur (M0.4 Bayes ızgarası).

Sorun: $p(x\mid y)$'yi öğrenmek. $d$ ikili öznitelik için $x$'in $2^d$ olası değeri vardır; tam bir tablo sınıf başına $2^d-1$ olasılık ister. $d=30$'da bir milyardan fazla — hiçbir veri kümesi buna yetmez.

### Naif varsayım {#naif}

:::tanım Koşullu bağımsızlık varsayımı
Sınıf bilindiğinde öznitelikler birbirinden bağımsızdır:
$$p(x_1,\dots,x_d\mid y)=\prod_{j=1}^dp(x_j\mid y).$$
Parametre sayısı $2^d-1$'den $d$'ye (ikili öznitelikte, sınıf başına) iner. Her $p(x_j\mid y)$ tek boyutlu bir dağılımdır ve **ayrı ayrı sayılarak** öğrenilir.
:::

**Örnek: spam.** Öznitelikler "e-postada şu kelime geçiyor mu?". Varsayım: "e-postanın spam olduğu bilindiğinde, 'ücretsiz' kelimesinin geçmesi 'kazandınız' kelimesinin geçip geçmediğinden bağımsızdır." Açıkça yanlıştır (ikisi birlikte geçme eğilimindedir) — ama sınıflandırma için şaşırtıcı derecede iyi çalışır.

**Neden "yanlış ama işe yarar"?** Sınıflandırma yalnız $\arg\max$'ın doğru olmasını ister; olasılıkların kendisinin doğru olması gerekmez. Bağımlı öznitelikler aynı kanıtı birden çok kez sayar ve posterior'u aşırı uçlara (0 ya da 1'e) iter, ama çoğu zaman **yönünü** değiştirmez (Domingos ve Pazzani 1997). Sonuç: iyi doğruluk, kötü **kalibrasyon**.

### Log uzayında karar {#log}

$$\log p(y\mid x)=\log p(y)+\sum_{j=1}^d\log p(x_j\mid y)-\log Z,\qquad Z=\sum_{y'}p(y')\prod_jp(x_j\mid y').$$
Çarpım yerine toplam iki sebeple: (1) yüzlerce küçük olasılığın çarpımı alttan taşar (M0.4); (2) toplam, doğrusal cebire dönüşür. Normalleştirme $\log Z$ bir **log-sum-exp**'tir (M1.5): sınıf skorları $s_y=\log p(y)+\sum_j\log p(x_j\mid y)$ ise $\log p(y\mid x)=s_y-\mathrm{LSE}(s)$.

**Gizli doğrusallık.** Kategorik/ikili öznitelikleri one-hot olarak kodlayınca $\sum_j\log p(x_j\mid y)=\phi(x)^\top\theta_y$: sınıf skoru özniteliklerin **doğrusal** bir fonksiyonudur. Yani Naive Bayes de, lojistik regresyon gibi, öznitelik uzayında doğrusal bir sınıflandırıcıdır; fark parametrelerin nasıl bulunduğundadır — sayarak (her öznitelik ayrı) ya da koşullu olabilirliği optimize ederek (birlikte).

### Parametre kestirimi: saymak {#sayma}

Hepsi M0.4'teki MLE'dir.

| tür | $p(x_j\mid y=c)$ | MLE | kullanım |
|---|---|---|---|
| kategorik | Kategorik($\theta_{jc}$) | $\hat\theta_{jcv}=\frac{n_{jcv}}{n_c}$ | tablo verisi |
| Bernoulli | Bernoulli($\theta_{jc}$) | $c$ sınıfında $x_j=1$ olanların oranı | kelime var/yok |
| multinomial | belge = $\sum_j x_j$ kelimelik torba | $\hat\theta_{cw}=\frac{\text{sayım}_{c,w}}{\sum_{w'}\text{sayım}_{c,w'}}$ | kelime sayımları |
| Gauss | $\mathcal N(\mu_{jc},\sigma^2_{jc})$ | sınıf içi ortalama ve varyans | sürekli öznitelik |

Önsel: $\hat p(y=c)=n_c/n$. **Eğitim tek geçişlik sayımdır:** iterasyon yok, optimizasyon yok, çok hızlı ve paralel. Vektörize hâli tek bir one-hot matris çarpımıdır: $Y\in\{0,1\}^{n\times C}$ etiketlerin one-hot'u, $X$ öznitelik sayımları ise sınıf-öznitelik sayım tablosu $Y^\top X$ $(C\times d)$ (modülün **c**–**d** aşamaları).

### Sıfır sorunu ve Laplace düzeltmesi {#laplace}

Eğitimde "ücretsiz" kelimesi hiç meşru e-postada geçmediyse MLE $p(\text{ücretsiz}\mid\text{meşru})=0$ verir. Bu kelimeyi içeren her yeni e-postanın meşru olma olasılığı, diğer **tüm** kanıtlardan bağımsız olarak **sıfır** olur: log uzayında $-\infty$. Tek bir görülmemiş kelime kararı tek başına belirler.

:::türetim Laplace (ek-α) düzeltmesi ve Dirichlet önseli
$V$ olası değer için her sayıma $\alpha>0$ ekle:
$$\hat p(x_j=v\mid y=c)=\frac{n_{jcv}+\alpha}{n_c+\alpha V}.$$
Hiç görülmemiş değer artık $\frac{\alpha}{n_c+\alpha V}>0$ alır; olasılıklar yine 1'e toplanır. **Bayesçi yorum:** kategorik parametreye Dirichlet($\alpha,\dots,\alpha$) önseli koyup **sonsal ortalamasını** almak tam olarak bu formülü verir; $\alpha=1$ düzgün önseldir ("her değeri bir kez görmüş gibi davran"), $\alpha<1$ (Lidstone) daha zayıf bir önseldir. M0.4'teki Beta–Bernoulli MAP ile aynı fikir. Veri arttıkça ($n_c\gg\alpha V$) düzeltmenin etkisi kaybolur.
**Seçim:** $\alpha$ bir hiperparametredir; çapraz doğrulamayla seçilir (modülün **e** aşamasında 20 değeri tek tensörle). Metinde $V$ (sözlük boyu) büyük olduğu için $\alpha=1$ nadir kelimelere fazla olasılık verebilir; $0.01$–$0.1$ sık kullanılır.
:::

### Gaussian Naive Bayes ve karar sınırları {#gauss}

Sürekli özniteliklerde her $p(x_j\mid y=c)=\mathcal N(\mu_{jc},\sigma^2_{jc})$. Sınıf skoru:
$$s_c(x)=\log\pi_c-\sum_j\Big[\frac{(x_j-\mu_{jc})^2}{2\sigma_{jc}^2}+\tfrac12\log(2\pi\sigma_{jc}^2)\Big].$$
İki sınıf arasındaki sınır $s_1(x)=s_0(x)$:

- **Varyanslar sınıfa göre farklıysa** $x_j^2$ terimleri kalır: sınır **kuadratiktir** (elips, hiperbol, parabol).
- **Varyanslar sınıflar arasında ortaksa** ($\sigma_{j0}=\sigma_{j1}$) $x_j^2$ terimleri sadeleşir:
$$s_1-s_0=\sum_j\frac{\mu_{j1}-\mu_{j0}}{\sigma_j^2}x_j+\text{sabit}=w^\top x+b.$$
Sınır **doğrusaldır** ve posterior $p(y=1\mid x)=\sigma(w^\top x+b)$ — tam olarak lojistik regresyonun biçimi!

:::ayrıntı Naive Bayes mi, lojistik regresyon mu? (Ng ve Jordan 2002)
Aynı model ailesi (doğrusal sınır), iki farklı kestirim: NB her parametreyi ayrı ayrı sayar (üretken), lojistik regresyon koşullu olabilirliği birlikte optimize eder (ayırt edici). Sonuç: (1) varsayım doğru değilse (öznitelikler ilişkiliyse) lojistik regresyonun **asimptotik hatası daha düşüktür** — yanlış varsayımı düzeltemeyen NB yanlı kalır. (2) NB parametrelerine **çok daha hızlı yakınsar**: gerekli örnek sayısı $d$ ile logaritmik ($O(\log d)$), lojistik regresyonda doğrusal ($O(d)$). Pratikte: öznitelik sayısına göre az veride NB, çok veride (ve varsayım bozuksa) lojistik regresyon kazanma eğilimindedir; öğrenme eğrileri kesişebilir. Bu bir eğilimdir, garanti değil: çok küçük örneklemlerde NB'nin varyans kestirimleri de gürültülüdür.
:::

Aşağıda sınıf içi korelasyonu artır (naif varsayımı boz), varyansları farklılaştır ve kalibrasyonun nasıl bozulduğunu gör.

::lab l1

İkinci deney Ng–Jordan karşılaştırmasını yüksek boyutta yapar: özniteliklerin yarısı bilgi taşır, diğer yarısı yalnız gürültüdür. Ortak bir gürültü faktörü eklenince bilgisiz öznitelikler o gürültüyü "ölçer"; en iyi sınıflandırıcı onları gürültüyü çıkarmak için kullanır — koşullu bağımsızlık varsayan NB bunu yapamaz.

::lab l2

### Metin sınıflandırma: kelime torbası {#metin}

Belge $\to$ kelime sayım vektörü $x\in\mathbb N^V$ (sıra atılır: **kelime torbası**). Multinomial NB:
$$\log p(y=c\mid x)=\log\pi_c+\sum_{w=1}^Vx_w\log\theta_{cw}+\text{sabit}=\log\pi_c+x^\top\log\theta_c.$$
Bir matris–vektör çarpımı: $V=50\,000$ kelimelik sözlükte bile milisaniyeler. Spam filtreleri, duygu analizi ve konu sınıflandırmasında yıllarca güçlü bir taban çizgisi oldu; bugün de az etiketli veride ve hızın önemli olduğu yerlerde hızlı bir ilk model ve **karşılaştırma tabanıdır** (Wang ve Manning 2012, "NBSVM").

**Pratik ayrıntılar:** durdurma kelimeleri ve çok nadir kelimeler genelde atılır; sayımlar yerine $\log(1+x)$ ya da TF-IDF ağırlıkları kullanmak sık iyileştirme sağlar; belge uzunluğu farkları multinomial modelde olasılıkları uçlara iter.

### Güçlü ve zayıf yanlar {#ozet-arti-eksi}

| artı | eksi |
|---|---|
| eğitim tek geçiş sayım; çok hızlı, akış (streaming) verisinde artımlı güncellenebilir | koşullu bağımsızlık varsayımı çoğu zaman yanlış |
| az veride iyi | olasılıklar kötü kalibre (aşırı güvenli); olasılık gerekiyorsa sonradan kalibre et |
| yüksek boyutta (metin) iyi ölçeklenir | öznitelik etkileşimlerini (XOR) yakalayamaz |
| eksik öznitelik: o terimi toplamdan çıkar | sürekli özniteliklerde dağılım varsayımı (Gauss) yanlış olabilir |

### Özet {#ozet}

| kavram | formül / kural |
|---|---|
| Bayes kararı | $\arg\max_yp(x\mid y)p(y)$ |
| naif varsayım | $p(x\mid y)=\prod_jp(x_j\mid y)$ |
| log skor | $s_y=\log p(y)+\sum_j\log p(x_j\mid y)$, normalleştirme LSE |
| kestirim | sınıf-öznitelik sayımları $Y^\top X$ |
| Laplace | $\frac{n+\alpha}{N+\alpha V}$ = Dirichlet sonsal ortalaması |
| Gauss NB | ortak varyans → doğrusal sınır (lojistik biçim), ayrı varyans → kuadratik |

**Kaynaklar:** Stanford CS229 ders notları, "Generative learning algorithms" (GDA, Naive Bayes, Laplace düzeltmesi); Murphy, *PML1* §9.3; Ng ve Jordan (2002), "On discriminative vs. generative classifiers: a comparison of logistic regression and naive Bayes"; Domingos ve Pazzani (1997), "On the optimality of the simple Bayesian classifier under zero-one loss"; Manning, Raghavan, Schütze, *Introduction to Information Retrieval*, Bölüm 13; Wang ve Manning (2012), "Baselines and bigrams".
