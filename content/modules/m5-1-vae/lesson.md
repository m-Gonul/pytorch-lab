### Üretken yapay zekânın dört yolu {#dort}

Şimdiye kadarki modellerin çoğu **ayırt edici**ydi: bir girdiye (görüntü, cümle) bir çıktı (sınıf, kutu, çeviri) atadılar. **Üretken** bir model ise verinin dağılımını öğrenir ve o dağılımdan **yeni örnekler** üretir: hiç var olmamış ama eğitim verisine benzeyen yüzler, cümleler, sesler. Ders slaytları 2026 itibarıyla dört ana yaklaşım sayar:

| yaklaşım | fikir | bu sitede |
|---|---|---|
| otoregresif | şimdiye kadarki token/piksellerden bir sonrakini üret: $p(x)=\prod_tp(x_t\mid x_{<t})$ | M4.7 (GPT), M4.16 |
| değişimsel otomatik kodlayıcı (VAE) | girdiyi bir gizil temsile eşle, gizil temsilden çıktıyı üret | **bu modül** |
| üretken çekişmeli ağ (GAN) | üreteç ve ayırt edici sıfır toplamlı bir oyun oynar | M5.2, M5.7 |
| difüzyon (DDPM) | gürültüden başlayıp gürültüyü adım adım gider | M5.3, M5.8 |

Dört yolun ortak sorunu şudur: yüksek boyutlu bir uzayda (ör. 64 × 64 × 3 = 12 288 boyut) **olasılık yoğunluğu** öğrenmek ve ondan örneklemek. Her yaklaşım bu zorluğu farklı yerden keser. Otoregresif modeller yoğunluğu tam hesaplar ama sırayla üretir. VAE'ler yoğunluğun bir alt sınırını en büyükler. GAN'lar yoğunluğu hiç yazmaz, yalnız örnek üretmeyi öğrenir. Difüzyon modelleri gürültü giderme adımlarından oluşan uzun bir zincir kurar.

### Otomatik kodlayıcıdan değişimsel otomatik kodlayıcıya {#ae}

Bir **otomatik kodlayıcı** (autoencoder, AE) iki parçadan oluşur. **Kodlayıcı** (tanıma ağı) girdiyi düşük boyutlu bir **gizil temsile** (kod) çevirir. **Kod çözücü** (üretici ağ) koddan girdiyi yeniden kurmaya çalışır. Kayıp yeniden yapılandırma hatasıdır. Kod boyutu girdiden küçükse (eksik tamamlanmış, undercomplete AE) ağ girdiyi olduğu gibi kopyalayamaz ve verinin en önemli özelliklerini öğrenmek zorunda kalır. Otomatik kodlayıcı ailesinin tamamı (yığılmış, konvolüsyonel, gürültü giderici, seyrek AE; ağırlık bağlama; denetimsiz ön eğitim) M5.6'da işleniyor. Burada yalnız VAE'ye giden yolu izliyoruz.

:::türetim Doğrusal otomatik kodlayıcı = PCA alt uzayı
Kodlayıcı $\mathbf Z=\mathbf X\mathbf W_e$ ($d\to k$), kod çözücü $\hat{\mathbf X}=\mathbf Z\mathbf W_d$, kayıp $\frac1n\|\hat{\mathbf X}-\mathbf X\|_F^2$ olsun; veri merkezlenmiş. $\hat{\mathbf X}=\mathbf X\mathbf W_e\mathbf W_d$ ve $\mathbf A=\mathbf W_e\mathbf W_d$'nin rankı en fazla $k$'dır. Dolayısıyla en iyi AE, $\mathbf X$'in rank-$k$'lık en iyi yaklaşımından daha iyi olamaz. Eckart–Young teoremine göre (M2.6) bu yaklaşım, kovaryansın ilk $k$ özvektörünün gerdiği alt uzaya izdüşümdür ve hata atılan özdeğerlerin toplamıdır. Bu alt sınıra $\mathbf W_e=\mathbf U_k\mathbf M$, $\mathbf W_d=\mathbf M^{-1}\mathbf U_k^\top$ ile ulaşılır; burada $\mathbf M$ herhangi bir tersinir matristir. Yani doğrusal AE PCA ile **aynı alt uzayı** öğrenir, ama eksenleri ortogonal ve sıralı olmak zorunda değildir (Baldi ve Hornik 1989). Aşama c bunu gradyan inişiyle doğruluyor: eğitilen AE, PCA optimumunun %1 yakınına iner.
:::

**Sıradan AE neden üretken değildir?** Bir AE'yi iki boyutlu gizil uzayla MNIST'te eğitip kodları çizince (slayt 17–18) rakam sınıfları ayrı bulutlara dağılır, ama bulutlar düzensizdir. Aralarında geniş boşluklar ve uzun kuyruklar vardır; uzayın hangi bölgesinin "geçerli" kodlar içerdiği bilinmez. Rastgele bir nokta seçip kod çözücüden geçirirsen çoğu zaman anlamsız bir görüntü çıkar. Üretim için kodların **bilinen ve örneklenebilir** bir dağılıma uyması gerekir. VAE tam olarak bunu zorlar.

**VAE** (Kingma ve Welling 2013; aynı dönemde bağımsız olarak Rezende, Mohamed ve Wierstra 2014) şimdiye kadarki otomatik kodlayıcılardan iki bakımdan farklıdır:
- **Olasılıksaldır:** çıktıları eğitimden sonra da kısmen rastlantıya bağlıdır. Gürültü giderici AE ise rastgeleliği yalnız eğitimde kullanır.
- **Üretkendir:** eğitim kümesinden örneklenmiş gibi görünen yeni örnekler üretebilir.

Kodlayıcı doğrudan bir kod üretmez. Bir **ortalama kod** $\boldsymbol\mu$ ve bir **standart sapma** $\boldsymbol\sigma$ üretir. Asıl kod bu ortalama ve standart sapmalı bir Gauss'tan rastgele örneklenir, sonra kod çözücü onu her zamanki gibi çözer. Eğitimde maliyet fonksiyonu kodları gizil uzayda yavaş yavaş kaydırır; sonunda kodlar bir Gauss nokta bulutu gibi görünür. Önemli sonucu şudur: eğitilmiş bir VAE ile yeni bir örnek üretmek çok kolaydır. Gauss dağılımından rastgele bir kod örneklenir ve kod çözücüden geçirilir.

<figure class="fig wide"><svg viewBox="0 0 640 288" role="img" aria-label="VAE: kodlayıcı x'ten ortalama ve log varyans üretir, z ortalama artı standart sapma çarpı dış gürültü olarak örneklenir, kod çözücü z'den x'i yeniden üretir"><defs><marker id="k1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="k2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><rect x="20" y="80" width="60" height="40" rx="5" class="soft-a"/><text x="50.0" y="104.0" text-anchor="middle" class="ta">x</text><line x1="80" y1="100" x2="110" y2="100" class="ink" marker-end="url(#k1)"/><rect x="112" y="70" width="110" height="60" rx="5" class="soft-a"/><text x="167.0" y="104.0" text-anchor="middle" class="sm">kodlayıcı q<tspan baseline-shift="-3" font-size="75%">φ</tspan></text><line x1="222" y1="88" x2="262" y2="70" class="ink" marker-end="url(#k1)"/><line x1="222" y1="112" x2="262" y2="130" class="ink" marker-end="url(#k1)"/><rect x="264" y="56" width="70" height="28" rx="5" class="soft-p"/><text x="299.0" y="74.0" text-anchor="middle" class="sm">μ(x)</text><rect x="264" y="116" width="70" height="28" rx="5" class="soft-p"/><text x="299.0" y="134.0" text-anchor="middle" class="sm">log σ²(x)</text><rect x="264" y="176" width="70" height="28" rx="5" class="soft-b"/><text x="299.0" y="194.0" text-anchor="middle" class="sm">ε ~ N(0, I)</text><circle cx="390" cy="100" r="20" class="soft-p"/><text x="390" y="104" text-anchor="middle" class="sm">z</text><line x1="334" y1="70" x2="372" y2="92" class="ink" marker-end="url(#k1)"/><line x1="334" y1="130" x2="372" y2="108" class="ink" marker-end="url(#k1)"/><line x1="334" y1="190" x2="378" y2="116" class="cb" marker-end="url(#k2)"/><text x="360" y="160" class="sm">z = μ + σ ⊙ ε</text><line x1="410" y1="100" x2="440" y2="100" class="ink" marker-end="url(#k1)"/><rect x="442" y="70" width="110" height="60" rx="5" class="soft-a"/><text x="497.0" y="104.0" text-anchor="middle" class="sm">kod çözücü p<tspan baseline-shift="-3" font-size="75%">θ</tspan></text><line x1="552" y1="100" x2="580" y2="100" class="ink" marker-end="url(#k1)"/><rect x="582" y="80" width="46" height="40" rx="5" class="soft-a"/><text x="605.0" y="104.0" text-anchor="middle" class="ta">x′</text><text x="20" y="232" class="sm">Gradyan μ ve σ üzerinden geçer (siyah oklar), rastgelelik ε dışarıdan gelir (turuncu ok): ∂z/∂μ = 1, ∂z/∂σ = ε.</text><text x="20" y="250" class="sm">Kayıp = −E<tspan baseline-shift="-3" font-size="75%">q</tspan>[log p<tspan baseline-shift="-3" font-size="75%">θ</tspan>(x|z)] (yeniden yapılandırma, x′ ile x arasında) + KL(q<tspan baseline-shift="-3" font-size="75%">φ</tspan>(z|x) ‖ N(0, I)) (gizil kayıp).</text><text x="20" y="276" class="sm">Üretim: z ~ N(0, I) → kod çözücü. Kodlayıcı yalnız eğitimde kullanılır.</text></svg><figcaption>Şekil 1 — VAE ve reparametrizasyon hilesi. Kodlayıcı doğrudan bir kod değil, bir Gauss dağılımının parametrelerini üretir. Kod, bu dağılımdan dışarıdan gelen standart bir gürültüyle örneklenir. Böylece örnekleme adımı, μ ve σ'ya göre türevlenebilir bir fonksiyona dönüşür.</figcaption></figure>

### Gizil değişkenli model ve KL ıraksaması {#gizil}

VAE'nin arkasındaki olasılık modeli şudur: önce bir gizil değişken önselden çekilir, $\mathbf z\sim p(\mathbf z)=\mathcal N(\mathbf 0,\mathbf I)$; sonra gözlem kod çözücünün tanımladığı dağılımdan üretilir, $\mathbf x\sim p_\theta(\mathbf x\mid\mathbf z)$. Bir verinin olasılığı tüm olası kodlar üzerinden toplamdır:
$$p_\theta(\mathbf x)=\int p_\theta(\mathbf x\mid\mathbf z)\,p(\mathbf z)\,d\mathbf z .$$
Kod çözücü bir sinir ağı olduğunda bu integral **hesaplanamaz**: $\mathbf z$'nin her değeri için ağı çalıştırmak gerekir. Gerçek posterior $p_\theta(\mathbf z\mid\mathbf x)=p_\theta(\mathbf x\mid\mathbf z)p(\mathbf z)/p_\theta(\mathbf x)$ da paydası yüzünden hesaplanamaz. Değişimsel çıkarım bu posterioru basit bir **yaklaşık dağılımla** değiştirir: kodlayıcının ürettiği $q_\phi(\mathbf z\mid\mathbf x)=\mathcal N(\boldsymbol\mu_\phi(\mathbf x),\mathrm{diag}\,\boldsymbol\sigma^2_\phi(\mathbf x))$. "Değişimsel" adı, bir dağılım ailesi içinde en iyisinin aranmasından gelir.

İki dağılımın ne kadar farklı olduğunu **Kullback–Leibler (KL) ıraksaması** ölçer:
$$\mathrm{KL}(Q\,\|\,P)=\sum_xQ(x)\log\frac{Q(x)}{P(x)}\quad\Big(\text{sürekli: }\int q\log\frac qp\Big).$$
Slaytın yorumuyla, $Q$ veriyi ya da gözlenen dağılımı, $P$ bir kuramı, modeli ya da yaklaşımı temsil eder. $\log_2$ ile KL, $Q$'dan gelen örnekleri $Q$ için en iyi kod yerine $P$ için en iyi kodla kodlamanın gerektirdiği ortalama **fazladan bit sayısıdır**. KL simetrik değildir ve üçgen eşitsizliğini sağlamaz; bu yüzden bir uzaklık (metrik) değil, bir ıraksamadır.

:::kanıt KL ≥ 0, eşitlik ancak $Q=P$ iken
$-\log$ dışbükey olduğu için Jensen eşitsizliği şunu verir:
$$\mathrm{KL}(Q\|P)=\mathbb E_Q\Big[-\log\frac{P(x)}{Q(x)}\Big]\ge-\log\mathbb E_Q\Big[\frac{P(x)}{Q(x)}\Big]=-\log\sum_xP(x)=0 .$$
Eşitlik ancak $P/Q$ sabitse, yani $P=Q$ iken sağlanır (Gibbs eşitsizliği).
:::

:::sezgi Hangi yön? İleri ve ters KL
$\mathrm{KL}(Q\|P)$'de beklenti $Q$ altında alınır. $Q$'nun olasılık verdiği yerde $P$ küçükse $\log(Q/P)$ büyür ve ceza ağırdır. $Q$'nun sıfır verdiği yer ise hiç cezalandırılmaz. Bu yüzden $\mathrm{KL}(q\|p)$'yi (ters KL) en küçükleyen tek tepeli bir $q$, çok tepeli bir $p$'nin **bir tepesine** yerleşir: tepe arayan bir çözümdür. $\mathrm{KL}(p\|q)$'yi (ileri KL) en küçükleyen $q$ ise $p$'nin her yerini örtmek zorundadır ve iki tepenin arasına da olasılık koyar: moment eşleyen bir çözümdür. VAE'nin amacı ters yönü kullanır ($\mathrm{KL}(q\|\text{posterior})$). Gerçek posterior çok tepeliyse kodlayıcı genellikle bir tepeyi seçer ve diğerlerini kaçırır.
:::

::lab l1

### ELBO: iki türetme {#elbo}

$\log p_\theta(\mathbf x)$'i doğrudan en büyükleyemiyoruz. Ama onun bir **alt sınırını** en büyükleyebiliriz: **kanıtın alt sınırı** (evidence lower bound, ELBO).

:::türetim Jensen yoluyla (ders slaytındaki türetme)
$$\log p(\mathbf x)=\log\int p(\mathbf x,\mathbf z)\,d\mathbf z=\log\int q(\mathbf z\mid\mathbf x)\frac{p(\mathbf x,\mathbf z)}{q(\mathbf z\mid\mathbf x)}d\mathbf z=\log\mathbb E_{q}\Big[\frac{p(\mathbf x,\mathbf z)}{q(\mathbf z\mid\mathbf x)}\Big].$$
$\log$ içbükey olduğu için Jensen eşitsizliği $\log\mathbb E[\cdot]\ge\mathbb E[\log\cdot]$ verir:
$$\log p(\mathbf x)\ \ge\ \mathbb E_q\Big[\log\frac{p(\mathbf x,\mathbf z)}{q(\mathbf z\mid\mathbf x)}\Big]=\mathbb E_q\Big[\log\frac{p(\mathbf x\mid\mathbf z)p(\mathbf z)}{q(\mathbf z\mid\mathbf x)}\Big]=\underbrace{\mathbb E_q[\log p(\mathbf x\mid\mathbf z)]}_{\text{yeniden yapılandırma}}-\underbrace{\mathrm{KL}\big(q(\mathbf z\mid\mathbf x)\,\|\,p(\mathbf z)\big)}_{\text{gizil kayıp}}.$$
$\log p(\mathbf x)$'i en büyüklemek yerine ELBO'yu en büyükleriz, yani negatif ELBO'yu en küçükleriz.
:::

:::türetim Kesin özdeşlik: boşluk nedir?
Jensen bir eşitsizlik verir ama boşluğun ne olduğunu söylemez. $p(\mathbf x,\mathbf z)=p(\mathbf z\mid\mathbf x)p(\mathbf x)$ yazalım:
$$\mathbb E_q\Big[\log\frac{p(\mathbf x,\mathbf z)}{q(\mathbf z\mid\mathbf x)}\Big]=\mathbb E_q\Big[\log p(\mathbf x)+\log\frac{p(\mathbf z\mid\mathbf x)}{q(\mathbf z\mid\mathbf x)}\Big]=\log p(\mathbf x)-\mathrm{KL}\big(q(\mathbf z\mid\mathbf x)\,\|\,p(\mathbf z\mid\mathbf x)\big).$$
Yani
$$\log p(\mathbf x)=\mathrm{ELBO}+\mathrm{KL}\big(q(\mathbf z\mid\mathbf x)\,\|\,p(\mathbf z\mid\mathbf x)\big)\ \ge\ \mathrm{ELBO}.$$
Boşluk, yaklaşık posteriorun gerçek posteriordan KL uzaklığıdır. ELBO'yu $\phi$'ye göre en büyüklemek bu boşluğu küçültür ($\log p(\mathbf x)$, $\phi$'ye bağlı değildir). $\theta$'ya göre en büyüklemek ise modeli iyileştirir. İkisi tek bir amaçta birleşir.
:::

<figure class="fig wide"><svg viewBox="0 0 640 238" role="img" aria-label="Dört farklı q için ELBO çubukları: gerçek posteriorda ELBO log p(x)'e eşit, diğerlerinde aradaki boşluk q ile gerçek posterior arasındaki KL"><line x1="492.1" y1="20" x2="492.1" y2="216" class="cq w2 dash"/><text x="496.1" y="18" class="tq">log p(x) = -1.921</text><text x="190" y="50" text-anchor="end" class="sm">q = gerçek posterior</text><rect x="200.0" y="34" width="292.1" height="22" class="ca-f"/><text x="204.0" y="49" class="sm" style="fill:#fff">ELBO -1.921</text><text x="204.0" y="70" class="sm">boşluk = KL(q ‖ p(z|x)) = 0.000</text><text x="190" y="94" text-anchor="end" class="sm">q = N(0.67, 1)</text><rect x="200.0" y="78" width="129.2" height="22" class="ca-f"/><rect x="329.2" y="78" width="162.9" height="22" class="cb-f"/><text x="204.0" y="93" class="sm" style="fill:#fff">ELBO -4.822</text><text x="204.0" y="114" class="sm">boşluk = KL(q ‖ p(z|x)) = 2.901</text><text x="190" y="138" text-anchor="end" class="sm">q = N(1.2, 0.05)</text><rect x="200.0" y="122" width="213.3" height="22" class="ca-f"/><rect x="413.3" y="122" width="78.9" height="22" class="cb-f"/><text x="204.0" y="137" class="sm" style="fill:#fff">ELBO -3.325</text><text x="204.0" y="158" class="sm">boşluk = KL(q ‖ p(z|x)) = 1.404</text><text x="190" y="182" text-anchor="end" class="sm">q = önsel N(0, 1)</text><rect x="200.0" y="166" width="16.8" height="22" class="ca-f"/><rect x="216.8" y="166" width="275.3" height="22" class="cb-f"/><text x="204.0" y="181" class="sm" style="fill:#fff">ELBO -6.822</text><text x="204.0" y="202" class="sm">boşluk = KL(q ‖ p(z|x)) = 4.901</text><text x="20" y="226" class="sm">Model: p(z) = N(0, 1), p(x|z) = N(2z, 0.5), x = 1.5. Gerçek posterior N(0.667, 0.111). Mavi: ELBO; turuncu: boşluk.</text></svg><figcaption>Şekil 2 — ELBO'nun boşluğu, q'nun gerçek posteriordan KL uzaklığıdır. Sayılar kapalı formdan hesaplandı (aşama c'deki doğrusal-Gauss oyuncak model). q gerçek posteriora eşitken sınır sıkıdır; q'yu önsele eşitlemek en büyük boşluğu verir.</figcaption></figure>

:::uyarı Kaynak notu: gösterim ve önsel
Slayt 46 kodlayıcının parametrelerini $\theta$, kod çözücününkini $\phi$ ile gösteriyor: $q_\theta(\mathbf z\mid\mathbf x)$, $p_\phi(\mathbf x\mid\mathbf z)$. Kingma ve Welling'in makalesi ve literatürün çoğu bunun **tersini** kullanır: kodlayıcı (tanıma modeli) $q_\phi$, kod çözücü (üretici model) $p_\theta$. Bu sayfa, aşamalar ve ödevler literatürün gösterimini izler. Ayrıca slayt 48'deki $p(\mathbf z)=\mathcal N(0,1)$, çok boyutlu gizil uzayda $\mathcal N(\mathbf 0,\mathbf I)$ olarak okunmalıdır: her boyut bağımsız standart normaldir.
:::

### Gauss KL'nin kapalı formu {#kl}

Önsel ve yaklaşık posterior ikisi de Gauss olduğu için gizil kayıp **kapalı formda** hesaplanır. Örneklemeye gerek kalmaz.

:::türetim Tek boyut, sonra köşegen
$q=\mathcal N(\mu,\sigma^2)$, $p=\mathcal N(0,1)$:
$$\log q(z)-\log p(z)=-\log\sigma-\frac{(z-\mu)^2}{2\sigma^2}+\frac{z^2}2 .$$
$q$ altında beklenti alalım: $\mathbb E_q[(z-\mu)^2]=\sigma^2$ ve $\mathbb E_q[z^2]=\mu^2+\sigma^2$. Buradan
$$\mathrm{KL}(q\|p)=-\log\sigma-\frac12+\frac{\mu^2+\sigma^2}2=\frac12\big(\mu^2+\sigma^2-1-\log\sigma^2\big).$$
Köşegen kovaryanslı $K$ boyutlu Gauss'ta boyutlar bağımsızdır ve KL'ler toplanır:
$$\text{gizil kayıp}=-\frac12\sum_{i=1}^K\big(1+\log\sigma_i^2-\sigma_i^2-\mu_i^2\big).$$
Slayt 50–51'deki formül tam olarak budur. Sıfır olması için her boyutta $\mu_i=0$ ve $\sigma_i=1$ gerekir.
:::

:::örnek El hesabı (modülün el hesabı alanları)
$\mu=1$, $\sigma=0.5$: $\mathrm{KL}=\frac12(1+0.25-1-\log0.25)=\frac12(0.25+1.386)=\mathbf{0.818}$. Gradyanlar: $\partial\mathrm{KL}/\partial\mu=\mu=\mathbf 1$; $\partial\mathrm{KL}/\partial\sigma=\sigma-1/\sigma=0.5-2=\mathbf{-1.5}$. $\gamma=\log\sigma^2$ parametrizasyonunda $\mathrm{KL}=\frac12(\mu^2+e^\gamma-1-\gamma)$ ve $\partial\mathrm{KL}/\partial\gamma=\frac12(e^\gamma-1)=\frac12(0.25-1)=\mathbf{-0.375}$. Yorum: $\sigma<1$ iken KL $\sigma$'yı büyütmek ister. Kodlayıcının çok emin (dar) olması cezalandırılır. $\mu$ sıfırdan uzaklaştıkça $\mu^2/2$ cezası gelir. İkisi birlikte gizil uzayı $\mathcal N(\mathbf 0,\mathbf I)$ etrafında tutar.
:::

Slayt 52'deki yaygın değişiklik: kodlayıcı $\sigma$ yerine $\gamma=\log\sigma^2$ üretir. Gizil kayıp $-\frac12\sum_i(1+\gamma_i-e^{\gamma_i}-\mu_i^2)$ olur. Bu yaklaşım sayısal olarak daha kararlıdır ve eğitimi hızlandırır. Nedeni: $\sigma>0$ kısıtı kendiliğinden sağlanır (her gerçek $\gamma$ geçerlidir), $\log\sigma^2$'yi hesaplamak için küçük bir sayının logaritmasını almak gerekmez ve çıktı katmanı aktivasyonsuz kalabilir.

### Reparametrizasyon hilesi {#reparam}

Yeniden yapılandırma terimi $\mathbb E_{q_\phi(\mathbf z\mid\mathbf x)}[\log p_\theta(\mathbf x\mid\mathbf z)]$ kapalı formda değildir; örnekle kestirilir. Ama beklenti **$\phi$'ye bağlı bir dağılım** altında alınıyor. "Bir örnek çek, kaybı hesapla, geri yay" yapınca gradyan örnekleme adımından geçemez: örnek, $\phi$'nin türevlenebilir bir fonksiyonu değildir. Slaytlar bu adımı atlıyor, ama VAE'nin çalışmasının anahtarı budur.

:::türetim Reparametrizasyon
Rastgeleliği parametrelerden ayıralım: $\boldsymbol\varepsilon\sim\mathcal N(\mathbf 0,\mathbf I)$ (parametresiz) ve $\mathbf z=\boldsymbol\mu_\phi(\mathbf x)+\boldsymbol\sigma_\phi(\mathbf x)\odot\boldsymbol\varepsilon$. O zaman $\mathbf z$ tam olarak $\mathcal N(\boldsymbol\mu,\mathrm{diag}\,\boldsymbol\sigma^2)$ dağılımlıdır ve
$$\nabla_\phi\,\mathbb E_{q_\phi}[f(\mathbf z)]=\nabla_\phi\,\mathbb E_{\boldsymbol\varepsilon}\big[f(\boldsymbol\mu_\phi+\boldsymbol\sigma_\phi\odot\boldsymbol\varepsilon)\big]=\mathbb E_{\boldsymbol\varepsilon}\big[\nabla_\phi f(\boldsymbol\mu_\phi+\boldsymbol\sigma_\phi\odot\boldsymbol\varepsilon)\big].$$
Türev ve beklenti yer değiştirebilir, çünkü artık beklentinin dağılımı $\phi$'ye bağlı değildir (koşul: $f$ $\mathbf z$'ye göre türevlenebilir ve yeterince düzgün). Tek bir $\boldsymbol\varepsilon$ örneği yansız bir gradyan kestirimi verir. Zincir kuralının parçaları $\partial z/\partial\mu=1$ ve $\partial z/\partial\sigma=\varepsilon$'dur. El hesabı: $\mu=1$, $\sigma=0.5$, $\varepsilon=0.4$ için $z=\mathbf{1.2}$, $\partial z/\partial\sigma=\mathbf{0.4}$.
:::

**Alternatif: skor fonksiyonu** (REINFORCE, likelihood-ratio). $\nabla_\phi\mathbb E_{q_\phi}[f]=\mathbb E_{q_\phi}[f(\mathbf z)\nabla_\phi\log q_\phi(\mathbf z)]$. Bu özdeşlik $f$'nin türevlenebilir olmasını gerektirmez; ayrık değişkenlerde ve pekiştirmeli öğrenmede (Faz 11) kullanılır. Ama $f$'nin **türevini** değil yalnız **değerini** kullandığı için genellikle çok daha gürültülüdür. Reparametrizasyon ise $f$'nin türevlenebilir olmasını şart koşar. Aşağıdaki deneyde iki kestiriciyi aynı problemde karşılaştır ve türevlenemeyen bir $f$'de reparametrizasyonun neden **yanlış** cevap verdiğini gör.

::lab l2

### Kayıp, eğitim ve örnekleme {#egitim}

Maliyet fonksiyonunun iki parçası vardır:
1. **Yeniden yapılandırma kaybı:** otomatik kodlayıcıyı girdisini yeniden üretmeye iter. Kod çözücünün çıktısına göre doğru $\mathbf x_i$'yi gözlemenin beklenen log-olabilirliği en büyüklenir, yani negatifi en küçüklenir.
2. **Gizil kayıp:** kodların basit bir Gauss'tan örneklenmiş gibi görünmesini sağlar. İstenen dağılım $p(\mathbf z)=\mathcal N(\mathbf 0,\mathbf I)$ ile kodlayıcının ürettiği $q_\phi(\mathbf z\mid\mathbf x_i)$ arasındaki KL en küçüklenir.

**Kod çözücü olabilirliği hangi kaybı verir?**
- **Bernoulli** (pikseller $[0,1]$'de): $-\log p(\mathbf x\mid\mathbf z)=$ ikili çapraz entropi.
- **Gauss**, sabit varyanslı: $p(\mathbf x\mid\mathbf z)=\mathcal N(\hat{\mathbf x},\sigma_x^2\mathbf I)$ ise $-\log p=\frac1{2\sigma_x^2}\|\mathbf x-\hat{\mathbf x}\|^2+\text{sabit}$. Kare hata.

Pratikteki "MSE + KL" kaybı, örtük olarak $\sigma_x^2=\frac12$ seçer. MSE'yi başka bir ağırlıkla toplamak, $\sigma_x$'i değiştirmekle, yani KL'nin göreli ağırlığını (aşağıdaki $\beta$) değiştirmekle aynıdır. Aşama d'deki `elbo_loss` örnek başına $\|\mathbf x-\hat{\mathbf x}\|^2+\beta\,\mathrm{KL}$'yi kullanıyor.

**Eğitim döngüsü:** mini-batch için kodlayıcıdan $\boldsymbol\mu$, $\log\boldsymbol\sigma^2$ al; $\boldsymbol\varepsilon$ çek, $\mathbf z$'yi kur; kod çözücüden $\hat{\mathbf x}$; kayıp; geri yayılım. Örnek başına tek $\boldsymbol\varepsilon$ yeterlidir: batch içindeki örnekler zaten gürültüyü ortalar.

**Örnekleme** (slayt 53): kodlayıcı atılır. $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$ çekilir ve kod çözücüden geçirilir. Slaytın ifadesiyle kod çözücü, $\mathcal N(\mathbf 0,\mathbf I)$'yı veri uzayındaki dağılıma taşıyan bir eşlemeyi öğrenir.

:::sezgi VAE örnekleri neden bulanıktır?
Gauss kod çözücüsünün en iyi çıktısı, aynı koda karşılık gelebilecek tüm görüntülerin **ortalamasıdır**. Kodlayıcının çıktısı gürültülü olduğu için ($\sigma>0$) bir kod birkaç farklı görüntüyle eşleşir. Bu görüntüler kenarların konumunda biraz farklıysa ortalamaları bulanık bir kenardır. GAN'lar (M5.2) bu ortalamayı almaz: ayırt edici bulanıklığı "sahte" diye cezalandırır. Bu yüzden ders slaytlarının dediği gibi GAN'lar birkaç yıl içinde VAE'leri gerçekçilikte geçti. Modern çözümler (VQ-VAE, Stable Diffusion'ın otomatik kodlayıcısı) VAE'yi algısal kayıp ve çekişmeli kayıpla birleştirir.
:::

### Gizil uzay: AE ve VAE, interpolasyon {#gizil-uzay}

Slayt 54, MNIST'te iki boyutlu gizil uzayla eğitilmiş bir AE ile bir VAE'nin kodlarını yan yana gösterir. AE'nin kodları düzensiz ve yayılmıştır. VAE'ninkiler orijin çevresinde, standart normale benzeyen sıkı bir bulutta toplanır ve sınıflar birbirine pürüzsüzce geçer. (Şekil görüntü olduğu için metne dökülemedi.)

**Anlamsal interpolasyon** (slayt 56–57). İki görüntüyü piksel düzeyinde interpole etmek iki görüntünün üst üste bindirilmiş hâline benzer. Bunun yerine **kod düzeyinde** interpole edebiliriz: iki görüntüyü kodlayıcıdan geçir, iki kodu interpole et, interpole edilmiş kodu kod çözücüden geçir. Sonuç sıradan bir Fashion MNIST görüntüsüne benzer, ama iki orijinal görüntünün arasında bir şeydir. Slayta göre bu, gizil uzay dağılımının Gauss olmaya zorlanması sayesinde mümkündür.

:::uyarı Kaynak notu: "kodlar Gauss bulutu gibi görünür" ve interpolasyon
Bu ifade yaklaşık olarak doğrudur ama iki inceliği vardır.
1. **Toplam posterior önsel değildir.** KL terimi her $\mathbf x$ için $q(\mathbf z\mid\mathbf x)$'i önsele yaklaştırır. Tüm veri üzerinden ortalaması olan toplam posterior $q(\mathbf z)=\mathbb E_{\mathbf x}q(\mathbf z\mid\mathbf x)$ ise önselle tam örtüşmez. Önselin yüksek olasılık verdiği ama hiçbir verinin kodlanmadığı "delikler" kalabilir. Bu deliklerden örneklenen kodlar kötü görüntüler verir (Rosca ve ark. 2018; çözüm önerileri: öğrenilen önseller, VampPrior).
2. **Doğrusal interpolasyon önselin dışına çıkar.** $d$ boyutlu standart normalde kütle yarıçapı $\approx\sqrt d$ olan ince bir **kabukta** yoğunlaşır. İki bağımsız kodun orta noktasının normu ise $\approx\sqrt{d/2}$'dir; kod çözücünün eğitimde nadiren gördüğü bir bölgedir. Küresel interpolasyon (slerp; White 2016) kabuk üzerinde kalır (ek x3).
:::

:::türetim Norm neden $\sqrt d$'de yoğunlaşır?
$\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$ için $\|\mathbf z\|^2=\sum_iz_i^2$. Her $z_i^2$'nin ortalaması 1, varyansı 2'dir. Toplamın ortalaması $d$, standart sapması $\sqrt{2d}$'dir; dolayısıyla $\|\mathbf z\|^2/d$'nin göreli yayılımı $\sqrt{2/d}\to0$. İki bağımsız örnek için $(\mathbf z_0+\mathbf z_1)/2\sim\mathcal N(\mathbf 0,\frac12\mathbf I)$ ve normu $\approx\sqrt{d/2}$, yani kabuğun yarıçapının $1/\sqrt2\approx0.71$'i. $d=128$'de $\|\mathbf z\|/\sqrt d$'nin standart sapması ≈ 0.06'dır; 0.71, ortalamadan beş standart sapma uzaktadır.
:::

<figure class="fig wide"><svg viewBox="0 0 640 254" role="img" aria-label="Standart normal örneklerin normunun karekök d'ye bölünmüş dağılımı; boyut arttıkça 1 çevresinde daralır, doğrusal interpolasyon orta noktası 0.71'e düşer"><line x1="60" y1="210" x2="420" y2="210" class="ink"/><line x1="60" y1="20" x2="60" y2="210" class="ink"/><text x="60.0" y="226" text-anchor="middle" class="sm">0</text><text x="150.0" y="226" text-anchor="middle" class="sm">0.5</text><text x="240.0" y="226" text-anchor="middle" class="sm">1</text><text x="330.0" y="226" text-anchor="middle" class="sm">1.5</text><text x="420.0" y="226" text-anchor="middle" class="sm">2</text><text x="240.0" y="244" text-anchor="middle" class="sm">‖z‖ / √d</text><path d="M61.8,209.4 L63.6,208.8 L65.4,208.2 L67.2,207.6 L69.0,207.0 L70.8,206.4 L72.6,205.8 L74.4,205.3 L76.2,204.7 L78.0,204.1 L79.8,203.5 L81.6,202.9 L83.4,202.4 L85.2,201.8 L87.0,201.3 L88.8,200.7 L90.6,200.2 L92.4,199.6 L94.2,199.1 L96.0,198.5 L97.8,198.0 L99.6,197.5 L101.4,197.0 L103.2,196.5 L105.0,196.0 L106.8,195.5 L108.6,195.0 L110.4,194.6 L112.2,194.1 L114.0,193.7 L115.8,193.2 L117.6,192.8 L119.4,192.4 L121.2,191.9 L123.0,191.5 L124.8,191.1 L126.6,190.8 L128.4,190.4 L130.2,190.0 L132.0,189.7 L133.8,189.3 L135.6,189.0 L137.4,188.7 L139.2,188.4 L141.0,188.1 L142.8,187.8 L144.6,187.5 L146.4,187.3 L148.2,187.0 L150.0,186.8 L151.8,186.6 L153.6,186.3 L155.4,186.1 L157.2,186.0 L159.0,185.8 L160.8,185.6 L162.6,185.4 L164.4,185.3 L166.2,185.2 L168.0,185.0 L169.8,184.9 L171.6,184.8 L173.4,184.7 L175.2,184.7 L177.0,184.6 L178.8,184.6 L180.6,184.5 L182.4,184.5 L184.2,184.4 L186.0,184.4 L187.8,184.4 L189.6,184.4 L191.4,184.5 L193.2,184.5 L195.0,184.5 L196.8,184.6 L198.6,184.6 L200.4,184.7 L202.2,184.8 L204.0,184.9 L205.8,184.9 L207.6,185.0 L209.4,185.2 L211.2,185.3 L213.0,185.4 L214.8,185.5 L216.6,185.7 L218.4,185.8 L220.2,186.0 L222.0,186.1 L223.8,186.3 L225.6,186.5 L227.4,186.7 L229.2,186.8 L231.0,187.0 L232.8,187.2 L234.6,187.4 L236.4,187.6 L238.2,187.9 L240.0,188.1 L241.8,188.3 L243.6,188.5 L245.4,188.7 L247.2,189.0 L249.0,189.2 L250.8,189.5 L252.6,189.7 L254.4,189.9 L256.2,190.2 L258.0,190.4 L259.8,190.7 L261.6,191.0 L263.4,191.2 L265.2,191.5 L267.0,191.7 L268.8,192.0 L270.6,192.3 L272.4,192.5 L274.2,192.8 L276.0,193.1 L277.8,193.3 L279.6,193.6 L281.4,193.8 L283.2,194.1 L285.0,194.4 L286.8,194.6 L288.6,194.9 L290.4,195.2 L292.2,195.4 L294.0,195.7 L295.8,196.0 L297.6,196.2 L299.4,196.5 L301.2,196.7 L303.0,197.0 L304.8,197.2 L306.6,197.5 L308.4,197.8 L310.2,198.0 L312.0,198.2 L313.8,198.5 L315.6,198.7 L317.4,199.0 L319.2,199.2 L321.0,199.4 L322.8,199.7 L324.6,199.9 L326.4,200.1 L328.2,200.4 L330.0,200.6 L331.8,200.8 L333.6,201.0 L335.4,201.2 L337.2,201.4 L339.0,201.6 L340.8,201.8 L342.6,202.0 L344.4,202.2 L346.2,202.4 L348.0,202.6 L349.8,202.8 L351.6,203.0 L353.4,203.2 L355.2,203.4 L357.0,203.5 L358.8,203.7 L360.6,203.9 L362.4,204.0 L364.2,204.2 L366.0,204.4 L367.8,204.5 L369.6,204.7 L371.4,204.8 L373.2,205.0 L375.0,205.1 L376.8,205.3 L378.6,205.4 L380.4,205.5 L382.2,205.7 L384.0,205.8 L385.8,205.9 L387.6,206.0 L389.4,206.2 L391.2,206.3 L393.0,206.4 L394.8,206.5 L396.6,206.6 L398.4,206.7 L400.2,206.8 L402.0,206.9 L403.8,207.0 L405.6,207.1 L407.4,207.2 L409.2,207.3 L411.0,207.4 L412.8,207.5 L414.6,207.6 L416.4,207.7 L418.2,207.7 L420.0,207.8" class="cp w2" fill="none"/><path d="M61.8,210.0 L63.6,210.0 L65.4,210.0 L67.2,210.0 L69.0,210.0 L70.8,210.0 L72.6,210.0 L74.4,210.0 L76.2,210.0 L78.0,210.0 L79.8,210.0 L81.6,210.0 L83.4,210.0 L85.2,210.0 L87.0,210.0 L88.8,210.0 L90.6,210.0 L92.4,210.0 L94.2,210.0 L96.0,210.0 L97.8,210.0 L99.6,210.0 L101.4,210.0 L103.2,210.0 L105.0,210.0 L106.8,210.0 L108.6,210.0 L110.4,210.0 L112.2,210.0 L114.0,210.0 L115.8,210.0 L117.6,210.0 L119.4,210.0 L121.2,210.0 L123.0,210.0 L124.8,210.0 L126.6,210.0 L128.4,210.0 L130.2,210.0 L132.0,209.9 L133.8,209.9 L135.6,209.9 L137.4,209.9 L139.2,209.8 L141.0,209.8 L142.8,209.7 L144.6,209.6 L146.4,209.5 L148.2,209.3 L150.0,209.2 L151.8,209.0 L153.6,208.7 L155.4,208.5 L157.2,208.1 L159.0,207.8 L160.8,207.3 L162.6,206.8 L164.4,206.2 L166.2,205.5 L168.0,204.8 L169.8,203.9 L171.6,203.0 L173.4,201.9 L175.2,200.7 L177.0,199.4 L178.8,198.1 L180.6,196.5 L182.4,194.9 L184.2,193.2 L186.0,191.3 L187.8,189.3 L189.6,187.3 L191.4,185.1 L193.2,182.9 L195.0,180.5 L196.8,178.2 L198.6,175.7 L200.4,173.3 L202.2,170.8 L204.0,168.3 L205.8,165.8 L207.6,163.4 L209.4,161.0 L211.2,158.7 L213.0,156.5 L214.8,154.4 L216.6,152.4 L218.4,150.5 L220.2,148.8 L222.0,147.3 L223.8,146.0 L225.6,144.9 L227.4,143.9 L229.2,143.2 L231.0,142.7 L232.8,142.4 L234.6,142.4 L236.4,142.5 L238.2,142.9 L240.0,143.4 L241.8,144.2 L243.6,145.2 L245.4,146.3 L247.2,147.6 L249.0,149.1 L250.8,150.7 L252.6,152.4 L254.4,154.2 L256.2,156.2 L258.0,158.2 L259.8,160.3 L261.6,162.4 L263.4,164.6 L265.2,166.8 L267.0,169.0 L268.8,171.2 L270.6,173.3 L272.4,175.5 L274.2,177.6 L276.0,179.6 L277.8,181.6 L279.6,183.6 L281.4,185.5 L283.2,187.3 L285.0,189.0 L286.8,190.6 L288.6,192.2 L290.4,193.7 L292.2,195.0 L294.0,196.4 L295.8,197.6 L297.6,198.7 L299.4,199.8 L301.2,200.8 L303.0,201.7 L304.8,202.5 L306.6,203.3 L308.4,204.0 L310.2,204.6 L312.0,205.2 L313.8,205.7 L315.6,206.2 L317.4,206.7 L319.2,207.1 L321.0,207.4 L322.8,207.7 L324.6,208.0 L326.4,208.3 L328.2,208.5 L330.0,208.7 L331.8,208.9 L333.6,209.0 L335.4,209.1 L337.2,209.3 L339.0,209.4 L340.8,209.5 L342.6,209.5 L344.4,209.6 L346.2,209.7 L348.0,209.7 L349.8,209.8 L351.6,209.8 L353.4,209.8 L355.2,209.9 L357.0,209.9 L358.8,209.9 L360.6,209.9 L362.4,209.9 L364.2,209.9 L366.0,209.9 L367.8,210.0 L369.6,210.0 L371.4,210.0 L373.2,210.0 L375.0,210.0 L376.8,210.0 L378.6,210.0 L380.4,210.0 L382.2,210.0 L384.0,210.0 L385.8,210.0 L387.6,210.0 L389.4,210.0 L391.2,210.0 L393.0,210.0 L394.8,210.0 L396.6,210.0 L398.4,210.0 L400.2,210.0 L402.0,210.0 L403.8,210.0 L405.6,210.0 L407.4,210.0 L409.2,210.0 L411.0,210.0 L412.8,210.0 L414.6,210.0 L416.4,210.0 L418.2,210.0 L420.0,210.0" class="ca w2" fill="none"/><path d="M61.8,210.0 L63.6,210.0 L65.4,210.0 L67.2,210.0 L69.0,210.0 L70.8,210.0 L72.6,210.0 L74.4,210.0 L76.2,210.0 L78.0,210.0 L79.8,210.0 L81.6,210.0 L83.4,210.0 L85.2,210.0 L87.0,210.0 L88.8,210.0 L90.6,210.0 L92.4,210.0 L94.2,210.0 L96.0,210.0 L97.8,210.0 L99.6,210.0 L101.4,210.0 L103.2,210.0 L105.0,210.0 L106.8,210.0 L108.6,210.0 L110.4,210.0 L112.2,210.0 L114.0,210.0 L115.8,210.0 L117.6,210.0 L119.4,210.0 L121.2,210.0 L123.0,210.0 L124.8,210.0 L126.6,210.0 L128.4,210.0 L130.2,210.0 L132.0,210.0 L133.8,210.0 L135.6,210.0 L137.4,210.0 L139.2,210.0 L141.0,210.0 L142.8,210.0 L144.6,210.0 L146.4,210.0 L148.2,210.0 L150.0,210.0 L151.8,210.0 L153.6,210.0 L155.4,210.0 L157.2,210.0 L159.0,210.0 L160.8,210.0 L162.6,210.0 L164.4,210.0 L166.2,210.0 L168.0,210.0 L169.8,210.0 L171.6,210.0 L173.4,210.0 L175.2,210.0 L177.0,210.0 L178.8,210.0 L180.6,210.0 L182.4,210.0 L184.2,210.0 L186.0,210.0 L187.8,210.0 L189.6,210.0 L191.4,210.0 L193.2,210.0 L195.0,210.0 L196.8,209.9 L198.6,209.8 L200.4,209.7 L202.2,209.5 L204.0,209.1 L205.8,208.4 L207.6,207.3 L209.4,205.5 L211.2,203.0 L213.0,199.3 L214.8,194.2 L216.6,187.4 L218.4,178.5 L220.2,167.4 L222.0,154.0 L223.8,138.4 L225.6,121.1 L227.4,102.6 L229.2,83.7 L231.0,65.6 L232.8,49.2 L234.6,35.6 L236.4,25.9 L238.2,20.5 L240.0,20.0 L241.8,24.3 L243.6,33.0 L245.4,45.4 L247.2,60.7 L249.0,77.9 L250.8,95.9 L252.6,113.8 L254.4,130.8 L256.2,146.4 L258.0,160.0 L259.8,171.7 L261.6,181.3 L263.4,188.9 L265.2,194.9 L267.0,199.4 L268.8,202.8 L270.6,205.2 L272.4,206.8 L274.2,208.0 L276.0,208.7 L277.8,209.2 L279.6,209.5 L281.4,209.7 L283.2,209.8 L285.0,209.9 L286.8,210.0 L288.6,210.0 L290.4,210.0 L292.2,210.0 L294.0,210.0 L295.8,210.0 L297.6,210.0 L299.4,210.0 L301.2,210.0 L303.0,210.0 L304.8,210.0 L306.6,210.0 L308.4,210.0 L310.2,210.0 L312.0,210.0 L313.8,210.0 L315.6,210.0 L317.4,210.0 L319.2,210.0 L321.0,210.0 L322.8,210.0 L324.6,210.0 L326.4,210.0 L328.2,210.0 L330.0,210.0 L331.8,210.0 L333.6,210.0 L335.4,210.0 L337.2,210.0 L339.0,210.0 L340.8,210.0 L342.6,210.0 L344.4,210.0 L346.2,210.0 L348.0,210.0 L349.8,210.0 L351.6,210.0 L353.4,210.0 L355.2,210.0 L357.0,210.0 L358.8,210.0 L360.6,210.0 L362.4,210.0 L364.2,210.0 L366.0,210.0 L367.8,210.0 L369.6,210.0 L371.4,210.0 L373.2,210.0 L375.0,210.0 L376.8,210.0 L378.6,210.0 L380.4,210.0 L382.2,210.0 L384.0,210.0 L385.8,210.0 L387.6,210.0 L389.4,210.0 L391.2,210.0 L393.0,210.0 L394.8,210.0 L396.6,210.0 L398.4,210.0 L400.2,210.0 L402.0,210.0 L403.8,210.0 L405.6,210.0 L407.4,210.0 L409.2,210.0 L411.0,210.0 L412.8,210.0 L414.6,210.0 L416.4,210.0 L418.2,210.0 L420.0,210.0" class="cb w2" fill="none"/><line x1="187.3" y1="20" x2="187.3" y2="210" class="cq dash"/><text x="183.3" y="30" text-anchor="end" class="tq">1/√2</text><line x1="440" y1="40" x2="462" y2="40" class="cp w2"/><text x="468" y="44" class="sm">d = 2</text><line x1="440" y1="62" x2="462" y2="62" class="ca w2"/><text x="468" y="66" class="sm">d = 16</text><line x1="440" y1="84" x2="462" y2="84" class="cb w2"/><text x="468" y="88" class="sm">d = 128</text><text x="440" y="130" class="sm">İki bağımsız örneğin</text><text x="440" y="146" class="sm">doğrusal orta noktası:</text><text x="440" y="162" class="sm">‖(z₀ + z₁)/2‖ ≈ √(d/2)</text><text x="440" y="178" class="sm">→ kabuğun içinde; d = 128 için</text><text x="440" y="194" class="sm">önsel neredeyse hiç örneklemez.</text></svg><figcaption>Şekil 3 — Yüksek boyutlu Gauss önselinde kütle ince bir kabukta toplanır: ‖z‖/√d dağılımı (ki dağılımı, kesin hesap) boyut arttıkça 1 çevresinde daralır. İki kodun doğrusal ortalaması bu kabuğun içine (≈ 1/√2) düşer; kod çözücü böyle kodları eğitimde nadiren görür. Küresel interpolasyon (slerp, ek x3) kabuk üzerinde kalır.</figcaption></figure>

### β-VAE ve posterior çöküşü {#beta}

ELBO'daki iki terimin dengesi ayarlanabilir: $\mathcal L_\beta=\mathbb E_q[\log p(\mathbf x\mid\mathbf z)]-\beta\,\mathrm{KL}(q\|p)$.
- $\beta>1$ (β-VAE; Higgins ve ark. 2017): gizil kodu önsele daha çok yaklaştırır. Bağımsız ve yorumlanabilir faktörler ("ayrışmış" temsiller) teşvik edilir, karşılığında yeniden yapılandırma bozulur.
- $\beta<1$: yeniden yapılandırma iyileşir, örnekler önselden uzaklaşır.

**Posterior çöküşü.** Bazı gizil boyutlarda (ya da hepsinde) kodlayıcı önselin aynısını üretir: $q(z_i\mid\mathbf x)=p(z_i)$, KL sıfır. Kod çözücü bu boyutları yok sayar. Özellikle güçlü kod çözücülerde (ör. otoregresif) model gizil kodu hiç kullanmadan veriyi açıklayabilir. O zaman KL terimi kodu kullanmamayı ödüllendirir.

:::türetim Doğrusal VAE'de çöküş eşiği
Doğrusal kodlayıcı ve kod çözücü, gözlem gürültüsü $\sigma^2$ olan Gauss olabilirlik ve $\beta$ ağırlıklı KL ile amacın optimumu kapalı formdadır; olasılıksal PCA'nın genelleşmesidir (Lucas ve ark. 2019, ek x2). Veri kovaryansının özdeğerleri $\lambda_1\ge\lambda_2\ge\dots$ olsun. Optimumda $i$. özyön ancak
$$\lambda_i>\beta\sigma^2$$
ise gizil boyut olarak kullanılır. Kod çözücünün o yöndeki sütun normu $\sqrt{\lambda_i-\beta\sigma^2}$, boyut başına KL $\frac12\log\frac{\lambda_i}{\beta\sigma^2}$'dir. Eşiği aşmayan yönler çöker: KL = 0. Sezgi: bir yönü kodlamanın KL "maliyeti", o yönün kod çözücüye kazandırdığı açıklanan varyansla tartılır. $\beta$ ya da gürültü büyüdükçe yalnız güçlü yönler bu maliyete değer.
:::

Uygulamadaki çareler:
- **KL ısıtması:** $\beta$'yı 0'dan 1'e yavaşça artırmak (Bowman ve ark. 2016).
- **Serbest bitler:** her boyutta KL'nin belli bir eşiğin altını cezalandırmamak (Kingma ve ark. 2016).
- **Daha zayıf kod çözücü.**

Aşağıdaki deneyde doğrusal VAE'nin analitik optimumunda $\beta$, gürültü, özdeğer spektrumu ve gizil boyut sayısıyla hangi boyutların çöktüğünü gör.

::lab l3

### Daha sıkı sınır: IWAE {#iwae}

ELBO'nun boşluğu $\mathrm{KL}(q\|\text{posterior})$'dur. Önem ağırlıklı otomatik kodlayıcı (IWAE; Burda, Grosse ve Salakhutdinov 2016) $K$ örnekle daha sıkı bir sınır kurar:
$$\mathcal L_K=\mathbb E_{\mathbf z_1..\mathbf z_K\sim q}\Big[\log\frac1K\sum_{k=1}^K\frac{p(\mathbf x,\mathbf z_k)}{q(\mathbf z_k\mid\mathbf x)}\Big],\qquad\mathcal L_1=\mathrm{ELBO}\le\mathcal L_K\le\mathcal L_{K+1}\le\log p(\mathbf x).$$
$K\to\infty$ iken önem örneklemesi kestiricisi $p(\mathbf x)$'e yakınsar ve sınır $\log p(\mathbf x)$'e çıkar. Pratikte $\log\frac1K\sum e^{\log w_k}$ sayısal kararlılık için `logsumexp` ile hesaplanır (ek x1, döngüsüz). IWAE eğitimde kodlayıcıya daha az "posterioru tam örtme" baskısı yapar. Ayrıca bir modelin $\log p(\mathbf x)$'ini **değerlendirmek** için standart araçtır: büyük $K$ ile kestirilen sınır, ELBO'dan daha dürüst bir karşılaştırma ölçütüdür.

### PyTorch'ta {#pytorch}

| kavram | PyTorch | not |
|---|---|---|
| kodlayıcı çıktısı | `h = enc(x); mu, logvar = h.chunk(2, dim=-1)` | ya da iki ayrı `Linear` |
| reparametrizasyon | `z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)` | `std = exp(logvar / 2)` |
| Gauss KL | `kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1)` | örnek başına; sonra `.mean()` |
| Bernoulli kayıp | `F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1)` | logitlerle kararlı |
| Gauss kayıp | `((x - xhat) ** 2).sum(-1)` | örtük $\sigma_x^2=\frac12$ |
| β-VAE | `loss = (rec + beta * kl).mean()` | |
| dağılım nesneleri | `D.Normal(mu, std)`; `q.rsample()`; `D.kl_divergence(q, p)` | `rsample` reparametrize eder, `sample` etmez |
| IWAE | `torch.logsumexp(logw, dim=0) - math.log(K)` | `logw (K, B)` |
| üretim | `with torch.no_grad(): x = dec(torch.randn(n, d_lat))` | |

### Özet {#ozet}

- Dört üretken yaklaşım: otoregresif, VAE, GAN, difüzyon. VAE olabilirliğin bir alt sınırını en büyükler.
- Doğrusal AE, PCA'nın alt uzayını öğrenir (Eckart–Young); sıradan AE'nin gizil uzayı düzensizdir ve örneklenemez.
- Gizil değişkenli model $p(\mathbf x)=\int p(\mathbf x\mid\mathbf z)p(\mathbf z)d\mathbf z$ hesaplanamaz. Kodlayıcı $q_\phi(\mathbf z\mid\mathbf x)=\mathcal N(\boldsymbol\mu,\mathrm{diag}\,\boldsymbol\sigma^2)$ posterioru yaklaşıklar.
- KL ≥ 0 ve asimetriktir. Ters KL tepe arar, ileri KL momentleri eşler; tek tepeye oturan $q$ için KL(q‖p) ≈ log(1/tepe ağırlığı).
- $\log p(\mathbf x)=\mathrm{ELBO}+\mathrm{KL}(q\|p(\mathbf z\mid\mathbf x))$; ELBO = yeniden yapılandırma − KL(q‖önsel).
- Gauss KL $=\frac12\sum(\mu^2+\sigma^2-1-\log\sigma^2)$; el örneği 0.818, $\partial/\partial\sigma=-1.5$, $\partial/\partial\log\sigma^2=-0.375$. $\log\sigma^2$ parametrizasyonu kararlıdır.
- Reparametrizasyon $\mathbf z=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\varepsilon$ gradyanı örneklemeden geçirir; $f$'nin türevlenebilir olmasını gerektirir. Skor fonksiyonu genel ama gürültülüdür.
- MSE + KL, $\sigma_x^2=\frac12$ demektir; ağırlıkları değiştirmek β'yı değiştirmektir. Bulanıklık, Gauss kod çözücünün ortalama almasından gelir.
- Toplam posterior önsel değildir (delikler); doğrusal interpolasyon kabuğun içine düşer (norm ≈ 0.71·√d); slerp kullan.
- β-VAE; posterior çöküşü. Doğrusal VAE'de $\lambda_i>\beta\sigma^2$ eşiği. Çareler: KL ısıtması, serbest bitler. IWAE: $\log p(\mathbf x)$'e $K$ ile yaklaşan sınır.
