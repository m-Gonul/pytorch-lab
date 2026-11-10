### Problem: başlatma yetmez {#problem}

M3.5'te varyansı koruyan başlatma (He, Glorot) ve doymayan aktivasyonlarla sinyalin katmanlar boyunca sönmesini ya da patlamasını **başlangıçta** engelledik. Ama başlatma yalnız $t=0$ anı için bir garantidir: eğitim ilerledikçe ağırlıklar değişir, her katmanın girdisinin dağılımı da değişir ve kararsız gradyanlar geri dönebilir. Ioffe ve Szegedy (2015) bu değişime **iç ortak değişken kayması** (internal covariate shift) dediler: bir katman, önceki katmanların her güncellemesinde "yeni bir veri dağılımına" uyum sağlamak zorunda kalır. Önerdikleri çözüm: her gizli katmana, girdilerini her adımda **yeniden standartlaştıran** bir işlem eklemek — **Batch Normalization** (BN).

:::ayrıntı Yan not: katman = hesap grafiğinde bir düğüm
Bir nöron katmanı yazılımda genelde iki blok olarak kurulur: doğrusal blok $XW+b$ ve ardından aktivasyon bloğu. Her işlemin ayrı bir düğüm olması ters mod türevi (M3.3) kolaylaştırır. BN, bu iki bloğun arasına (ya da aktivasyondan sonra) yerleştirilen üçüncü bir bloktur. Çalıştırma sırasında bu bloklar çoğu zaman birleştirilir (füzyon; bu modülde BN'i doğrusal katmana katlayacağız, M6.5'te genel hâlini göreceğiz).
:::

"İç ortak değişken kayması" açıklaması sonradan tartışmalı çıktı (aşağıda **Neden işe yarıyor?**); ama algoritma ve etkileri sağlamdır.

### Algoritma: iki adım {#algoritma}

BN iki adımda çalışır: (1) girdiyi ortalaması 0, standart sapması 1 olacak biçimde dönüştür; (2) sonucu, katmanın **öğrenerek seçtiği** yeni bir ortalama ve standart sapmaya taşı. Birinci adımın istatistikleri **o anki mini-batch'ten** hesaplanır — adı buradan gelir.

:::tanım Batch Normalization (eğitim modu)
Bir mini-batch $B=\{x^{(1)},\dots,x^{(m_B)}\}$, her $x^{(i)}\in\mathbb R^d$ (bir katmanın $d$ girdisi). Tüm işlemler **özellik başına** (eleman eleman) yapılır:
$$\mu_B=\frac1{m_B}\sum_{i=1}^{m_B}x^{(i)},\qquad \sigma_B^2=\frac1{m_B}\sum_{i=1}^{m_B}\big(x^{(i)}-\mu_B\big)^2,$$
$$\hat x^{(i)}=\frac{x^{(i)}-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}},\qquad z^{(i)}=\gamma\otimes\hat x^{(i)}+\beta .$$
- $\mu_B,\sigma_B^2\in\mathbb R^d$: batch üzerinden, her girdi düğümü için ayrı ortalama ve varyans. Varyans **yanlı** ($\frac1{m_B}$) tahmindir.
- $\varepsilon$: sıfıra bölmeyi önleyen küçük sabit (PyTorch $10^{-5}$, Keras $10^{-3}$).
- $\gamma,\beta\in\mathbb R^d$: çıktı ölçeği ve kayması, geri yayılımla öğrenilir. $\otimes$ eleman eleman çarpım.
- $z^{(i)}$: BN'in çıktısı — girdinin yeniden ölçeklenmiş ve kaydırılmış hâli.
:::

**Neden $\gamma,\beta$?** Yalnız standartlaştırmak katmanın ifade gücünü kısıtlar: örneğin bir sigmoid'in girdisini hep $\mathcal N(0,1)$ civarına zorlamak onu neredeyse doğrusal bölgesine hapseder. $\gamma=\sqrt{\sigma_B^2+\varepsilon}$, $\beta=\mu_B$ seçilirse BN özdeşlik olur; yani BN ağın temsil edebileceği fonksiyonları **daraltmaz**, yalnız parametrelemeyi değiştirir. Başlangıç değerleri $\gamma=1$, $\beta=0$.

:::örnek Elle bir BN adımı
Tek özellik, batch $x=[1,2,3]$, $\varepsilon=0$, $\gamma=2$, $\beta=1$: $\mu_B=2$, $\sigma_B^2=\frac{1+0+1}3=\frac23$, $\sigma_B=0.8165$; $\hat x=[-1.2247,\,0,\,1.2247]$; $z=[-1.449,\,1,\,3.449]$. $\hat x$'in ortalaması 0, (yanlı) varyansı tam 1; $z$'nin ortalaması $\beta=1$, std'si $\gamma=2$.
:::

**Nereye konur?** Ioffe ve Szegedy BN'i doğrusal katmanla aktivasyonun **arasına** koydular: $a=\phi(\mathrm{BN}(XW))$. Bu durumda doğrusal katmanın biası gereksizdir: BN ortalamayı çıkarır ($b$ yok olur) ve $\beta$ onun işini görür — PyTorch'ta `nn.Linear(n, m, bias=False)` ya da `nn.Conv2d(..., bias=False)` yaz. Aktivasyondan **sonra** koymak da yaygındır ve bazı görevlerde daha iyi sonuç verir; tek bir doğru yer yoktur, ikisini de deneyebilirsin.

**İlk katman olarak BN.** Ağın en başına bir BN koyarsan girdiyi kendin standartlaştırmana çoğu zaman gerek kalmaz: BN bunu yaklaşık olarak yapar (yaklaşık, çünkü bir anda yalnız bir batch'e bakar; ayrıca $\gamma,\beta$ ile yeniden ölçekleyebilir).

### Eğitim ve çıkarım: iki farklı fonksiyon {#test}

Eğitimde istatistikler batch'ten gelir. Peki çıkarımda? Çoğu zaman **tek bir örnek** için tahmin isteriz (batch boyu 1): ortalama ve std hesaplanamaz. Bir batch olsa bile çok küçük olabilir ya da örnekleri bağımsız ve özdeş dağılımlı olmayabilir (aynı videonun ardışık kareleri gibi) — batch istatistikleri güvenilmez olur. Çıkarımda çıktı, batch'te başka hangi örneklerin olduğuna bağlı olmamalıdır.

<figure class="fig wide"><svg viewBox="0 0 680 190" role="img" aria-label="BatchNorm eğitim ve çıkarım yolları: eğitimde batch istatistikleri kullanılır ve kayan ortalamalar güncellenir; çıkarımda sabit koşu istatistikleri kullanılır"><defs><marker id="bn1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="bn2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cq-f"/></marker></defs><text x="10" y="22" class="ta">eğitim</text><rect x="10" y="32" width="120" height="46" rx="6" class="ink"/><text x="70.0" y="51.5" text-anchor="middle">mini-batch</text><text x="70.0" y="66.5" text-anchor="middle">x⁽¹⁾ … x⁽ᵐ⁾</text><rect x="170" y="32" width="140" height="46" rx="6" class="ca"/><text x="240.0" y="51.5" text-anchor="middle">μ<tspan baseline-shift="sub" font-size="10">B</tspan>, σ<tspan baseline-shift="sub" font-size="10">B</tspan><tspan baseline-shift="super" font-size="10">2</tspan></text><text x="240.0" y="66.5" text-anchor="middle">(batch üzerinden)</text><rect x="350" y="32" width="150" height="46" rx="6" class="ink"/><text x="425.0" y="51.5" text-anchor="middle">x<tspan dx="-0.42em" dy="-0.2em">ˆ</tspan><tspan dx="0.06em" dy="0.2em"> = (x − μ<tspan baseline-shift="sub" font-size="10">B</tspan>)</tspan></text><text x="425.0" y="66.5" text-anchor="middle">/ √(σ<tspan baseline-shift="sub" font-size="10">B</tspan><tspan baseline-shift="super" font-size="10">2</tspan> + ε)</text><rect x="540" y="32" width="120" height="46" rx="6" class="cp"/><text x="600.0" y="59.0" text-anchor="middle">z = γ ⊗ x<tspan dx="-0.42em" dy="-0.2em">ˆ</tspan><tspan dx="0.06em" dy="0.2em"> + β</tspan></text><line x1="130" y1="55" x2="168" y2="55" class="ink" marker-end="url(#bn1)"/><line x1="310" y1="55" x2="348" y2="55" class="ink" marker-end="url(#bn1)"/><line x1="500" y1="55" x2="538" y2="55" class="ink" marker-end="url(#bn1)"/><line x1="240" y1="78" x2="240" y2="128" class="cq dash" marker-end="url(#bn2)"/><text x="248" y="106" class="sm">kayan ortalama (EMA) ile güncelle</text><text x="10" y="124" class="tb">çıkarım</text><rect x="10" y="132" width="120" height="46" rx="6" class="ink"/><text x="70.0" y="151.5" text-anchor="middle">tek örnek x</text><text x="70.0" y="166.5" text-anchor="middle">(ya da batch)</text><rect x="170" y="132" width="140" height="46" rx="6" class="cq"/><text x="240.0" y="151.5" text-anchor="middle">μ, σ² (koşu</text><text x="240.0" y="166.5" text-anchor="middle">istatistikleri — sabit)</text><rect x="350" y="132" width="150" height="46" rx="6" class="ink"/><text x="425.0" y="151.5" text-anchor="middle">x<tspan dx="-0.42em" dy="-0.2em">ˆ</tspan><tspan dx="0.06em" dy="0.2em"> = (x − μ)</tspan></text><text x="425.0" y="166.5" text-anchor="middle">/ √(σ² + ε)</text><rect x="540" y="132" width="120" height="46" rx="6" class="cp"/><text x="600.0" y="159.0" text-anchor="middle">z = γ ⊗ x<tspan dx="-0.42em" dy="-0.2em">ˆ</tspan><tspan dx="0.06em" dy="0.2em"> + β</tspan></text><line x1="130" y1="155" x2="168" y2="155" class="ink" marker-end="url(#bn1)"/><line x1="310" y1="155" x2="348" y2="155" class="ink" marker-end="url(#bn1)"/><line x1="500" y1="155" x2="538" y2="155" class="ink" marker-end="url(#bn1)"/><text x="600" y="100" text-anchor="middle" class="tp">γ, β: geri yayılımla</text><text x="600" y="116" text-anchor="middle" class="sm">öğrenilir, iki yolda aynı</text></svg><figcaption>Şekil 1 — BatchNorm iki farklı fonksiyondur. Eğitimde ortalama ve varyans o anki batch'ten hesaplanır (çıktı batch'teki diğer örneklere bağlıdır) ve kayan ortalamalar güncellenir; çıkarımda o kayan ortalamalar sabit olarak kullanılır (çıktı deterministik, örnek başına). Katman başına 4 vektör: γ, β öğrenilir; μ, σ² veriden tahmin edilir.</figcaption></figure>

**Çözüm 1 — eğitim sonunda bir geçiş:** eğitim bittikten sonra tüm eğitim setini ağdan geçir, her BN katmanının girdisinin ortalamasını ve std'sini hesapla, çıkarımda bunları kullan. Doğru ama pahalı (ve katmanlar sırayla hesaplanmalı: bir katmanın istatistiği öncekilerin **son** istatistiklerine bağlı).

**Çözüm 2 — kayan ortalama (pratikte kullanılan):** eğitim boyunca batch istatistiklerinin üstel kayan ortalamasını (EMA, M3.4) tut. Böylece her BN katmanında **dört** parametre vektörü olur:

| vektör | anlamı | nasıl bulunur |
|---|---|---|
| $\gamma$ | çıktı ölçeği | geri yayılım |
| $\beta$ | çıktı kayması | geri yayılım |
| $\mu$ | "son" girdi ortalaması | EMA ile tahmin |
| $\sigma^2$ | "son" girdi varyansı | EMA ile tahmin |

$\mu$ ve $\sigma^2$ eğitim sırasında tahmin edilir ama **yalnız eğitimden sonra** kullanılır. PyTorch'ta bunlar öğrenilen parametre değil **buffer**'dır (`register_buffer`, d aşaması): `state_dict`'e kaydedilir, `.to(device)` ile taşınır, ama optimizer onları görmez.

:::ayrıntı Kütüphane konvansiyonları — dikkatli oku
- **PyTorch** (`momentum=0.1`): $\text{running\_mean}\leftarrow(1-0.1)\,\text{running\_mean}+0.1\,\mu_B$. Yani `momentum` **yeni** gözlemin ağırlığıdır (M3.4'teki $\beta$ momentumunun tersi).
- **Keras** (`momentum=0.99`): $\text{moving}\leftarrow0.99\,\text{moving}+0.01\,\mu_B$ — aynı kelime, ters anlam. Bir modeli taşırken $0.1\leftrightarrow0.99$ değil $0.1\leftrightarrow0.9$ eşleşir.
- **Varyans:** normalize ederken yanlı ($\frac1{m}$) varyans kullanılır, ama PyTorch `running_var`'ı **yansız** ($\frac1{m-1}$) batch varyansıyla günceller. Orijinal makale de çıkarım için $\mathrm{Var}[x]=\frac{m}{m-1}\mathbb E_B[\sigma_B^2]$ önerir.
- `momentum=None`: üstel değil, kümülatif (eşit ağırlıklı) ortalama — "Çözüm 1"in eğitim sırasındaki karşılığı.
- `track_running_stats=False`: hiç koşu istatistiği tutulmaz, çıkarımda da batch istatistiği kullanılır.
:::

:::uyarı Klasik BN hataları
- **`model.eval()`'i unutmak:** değerlendirmede batch istatistikleri kullanılır, sonuçlar batch bileşimine göre değişir; ayrıca koşu istatistikleri test verisiyle güncellenir (sızıntı).
- **Eğitimde batch boyu 1:** `BatchNorm1d` hata verir (*Expected more than 1 value per channel when training*): tek örneğin varyansı 0'dır.
- **İnce ayarda (fine-tuning) küçük batch:** önceden eğitilmiş ağın BN katmanlarını `eval()` modunda dondurmak çoğu zaman daha iyidir; aksi hâlde küçük ve gürültülü batch'ler iyi koşu istatistiklerini bozar.
- **Ağırlıkları sonradan değiştirmek** (ağırlık ortalama — SWA, budama): eski koşu istatistikleri artık yeni ağa uymaz; bir eğitim geçişiyle yeniden hesapla (`torch.optim.swa_utils.update_bn`).
:::

Eğitimde bir örneğin BN çıktısı batch'teki **diğer** örneklere bağlıdır. Aşağıda bunu tek bir özellik için izle: aynı $x$ değeri farklı batch'lerde farklı yerlere eşlenir; batch boyu küçüldükçe bu gürültü büyür.

::lab l1

Bu batch gürültüsü bir **düzenlileştirici**dir (her adımda örneklere biraz farklı bir dönüşüm uygulanır, dropout'a benzer; M3.7). Aynı gürültü küçük batch'lerde zarara dönüşür; batch boyu 2'de normalize değer yalnız $\pm1$ olabilir.

### Neden işe yarıyor? {#neden}

Ioffe ve Szegedy, denedikleri **tüm** derin ağlarda büyük iyileşme gösterdiler:

- **Sönen gradyanlar** o kadar azaldı ki tanh ve hatta sigmoid gibi doyan aktivasyonları kullanabildiler.
- Ağlar **başlatmaya** çok daha az duyarlı oldu.
- Çok daha **büyük öğrenme oranları** kullanabildiler; eğitim belirgin biçimde hızlandı.
- BN bir **düzenlileştirici** gibi davrandı ve dropout gibi diğer tekniklere ihtiyacı azalttı.

Makaleden: dönemin en iyi görüntü sınıflandırma modeline uygulandığında BN **aynı doğruluğa 14 kat daha az eğitim adımında** ulaştı ve orijinal modeli belirgin farkla geçti; BN'li ağlardan oluşan bir topluluk ImageNet'te **%4.9 top-5 doğrulama hatasına** (%4.8 test) indi — insan değerlendiricilerin doğruluğunun ötesine.

**Ama neden?** "İç ortak değişken kayması"nı azalttığı iddiası doğrudan sınandığında zayıf çıktı: Santurkar ve ark. (2018), BN'den sonra her adımda kasıtlı olarak **rastgele** dağılım kayması ekleyince bile BN'li ağın iyi eğitildiğini gösterdi; ölçtükleri kayma da BN'li ağlarda daha az değildi. Önerdikleri açıklama: BN kayıp yüzeyini ve gradyanı **daha pürüzsüz** (daha küçük Lipschitz sabitli) yapar; bu da büyük adımlara izin verir. Bugün en sağlam bilinen mekanizmalardan biri aşağıdaki **ölçek değişmezliği**dir.

:::türetim Ölçek değişmezliği ve etkin öğrenme oranı
$z=\mathrm{BN}(Wx)$ olsun (bias yok). Ağırlıkları bir $c>0$ ile ölçekle: $Wx\to cWx$, batch ortalaması $c\mu$, std $c\sigma$ olur ($\varepsilon$'u ihmal edersek), dolayısıyla
$$\mathrm{BN}(cWx)=\frac{cWx-c\mu}{c\sigma}=\mathrm{BN}(Wx).$$
Kayıp $W$'nin ölçeğine **bağlı değil**: $L(cW)=L(W)$. İki sonucu:
1. Bu eşitliği $W$'ye göre türet: $c\,\nabla L(cW)=\nabla L(W)$, yani $\nabla L(cW)=\frac1c\nabla L(W)$. Büyük ağırlıklar **küçük gradyan** alır — başlatma ölçeği ne olursa olsun bir patlama kendini sınırlar.
2. $L(cW)$'yi $c$'ye göre $c=1$'de türet: $\langle W,\nabla L(W)\rangle=0$. Gradyan $W$'ye **diktir**; SGD adımı $\|W\|$'yi yalnız büyütebilir (Pisagor: $\|W-\eta g\|^2=\|W\|^2+\eta^2\|g\|^2$).

Kaybı yalnız $W$'nin **yönü** belirlediğine göre önemli olan yöndeki değişimdir: yön $W/\|W\|$'nin bir adımda dönüşü yaklaşık $\eta\|g\|/\|W\|$, ve $\|g\|\propto1/\|W\|$ olduğundan **etkin öğrenme oranı** $\approx\eta/\|W\|^2$'dir. Ağırlık çürümesi (weight decay) $\|W\|$'yi küçülterek etkin öğrenme oranını **artırır** — BN'li ağlarda L2'nin rolü düzenlileştirme değil, bu dolaylı öğrenme oranı kontrolüdür (x3 ekinde ölçüyorsun).
:::

Aşağıdaki deneyde 20 katmanlı, kasıtlı olarak kötü başlatılmış bir ağda normalizasyonun etkisini katman katman izle. Başlatma çarpanını değiştirince normalizasyonsuz ağda aktivasyonlar üstel olarak söner ya da patlar; normalizasyonla std her katmanda aynı kalır ve gradyan normu tam $1/c$ ile ölçeklenir (yukarıdaki 1. sonuç).

::lab l2

:::uyarı Dürüst not: düz derin BN ağlarında gradyan patlaması
Deneyde BatchNorm seçiliyken "ilk/son" oranına bak: aktivasyonlar her katmanda düzgün ölçekli olduğu hâlde ilk katmanların gradyan normu son katmanınkinden onlarca kat büyük (20 katmanda ≈ 40–60×, 40 katmanda binlerce kat). Bu bir hata değil; Yang ve ark. (2019) artık bağlantısız derin BN ağlarında başlangıçta gradyanların derinlikle **üstel olarak patladığını** ve bunun başlatma ayarıyla giderilemediğini kuramsal olarak gösterdi. Pratikte BN hemen her zaman **artık bağlantılarla** (ResNet, M4.2) birlikte kullanılır. LayerNorm ve RMSNorm'da (örnek başına normalizasyon) bu dengesizlik deneyde çok daha ılımlıdır.
:::

### Geri geçiş: türetim {#geri}

BN'in geri geçişini yazmak (b ve c aşamaları) çok yollu zincir kuralının (M0.3) en öğretici örneklerinden biridir: $\mu_B$ ve $\sigma_B$ batch'teki **tüm** örneklere bağlıdır, dolayısıyla bir $x_k$'yı değiştirmek **tüm** $\hat x_i$'leri değiştirir. Tek bir özellik için yaz (her özellik bağımsız), $N=m_B$, $\sigma=\sqrt{\sigma_B^2+\varepsilon}$.

:::türetim $\partial L/\partial x$
Önce $\gamma,\beta$: $z_i=\gamma\hat x_i+\beta$ ⇒ $\frac{\partial L}{\partial\gamma}=\sum_i dz_i\,\hat x_i$, $\frac{\partial L}{\partial\beta}=\sum_i dz_i$, $d\hat x_i=\gamma\,dz_i$ (burada $dz_i=\partial L/\partial z_i$).

Ara türevler:
- $\dfrac{\partial\mu}{\partial x_k}=\dfrac1N$.
- $\dfrac{\partial\sigma_B^2}{\partial x_k}=\dfrac2N\sum_j(x_j-\mu)\Big(\delta_{jk}-\dfrac1N\Big)=\dfrac2N(x_k-\mu)$, çünkü $\sum_j(x_j-\mu)=0$.
- $\dfrac{\partial\sigma}{\partial x_k}=\dfrac1{2\sigma}\cdot\dfrac2N(x_k-\mu)=\dfrac{\hat x_k}{N}$ ($\varepsilon>0$ iken de tam doğru).

Bölüm kuralıyla:
$$\frac{\partial\hat x_i}{\partial x_k}=\frac{\delta_{ik}-\frac1N}{\sigma}-\frac{x_i-\mu}{\sigma^2}\cdot\frac{\hat x_k}N=\frac1\sigma\Big(\delta_{ik}-\frac1N-\frac{\hat x_i\hat x_k}N\Big).$$
Tüm yolları topla, $dx_k=\sum_i d\hat x_i\,\frac{\partial\hat x_i}{\partial x_k}$:
$$\boxed{dx_k=\frac1{N\sigma}\Big(N\,d\hat x_k-\sum_i d\hat x_i-\hat x_k\sum_i d\hat x_i\,\hat x_i\Big)}$$
:::

**Yorum.** Parantezdeki ilk terim "istatistikler sabitmiş gibi" olan yoldur; ikinci terim ortalama üzerinden, üçüncüsü varyans üzerinden gelen dolaylı yollardır. $\varepsilon=0$ iken sonuç $\sum_k dx_k=0$ ve $\sum_k dx_k\hat x_k=0$ sağlar: gradyan, normalizasyonun yok ettiği iki yönden (tüm batch'i kaydırmak, tüm batch'i ölçeklemek) **arındırılmıştır** — o yönlerde girdiyi değiştirmek çıktıyı değiştirmez, dolayısıyla gradyan o yönlerde sıfır olmalıdır. Hesap grafiğini adım adım geri yürümek (CS231n `batchnorm_backward`) aynı sonucu verir ama yavaştır; kutulu formül tek geçişte hesaplanır (`batchnorm_backward_alt`, c aşaması).

**Sayısal kontrol** (hand alıştırması): $x=[1,2,3]$, $\gamma=2$, $dz=[1,0,0]$ ⇒ $d\hat x=[2,0,0]$, $\sum d\hat x=2$, $\sum d\hat x\hat x=-2.449$; $dx_1=\frac{1}{3\cdot0.8165}\big(6-2-(-1.2247)(-2.449)\big)=0.408$, $dx_2=-0.816$, $dx_3=0.408$ — toplamları 0.

**LayerNorm ve RMSNorm** için formül aynıdır, yalnız toplamlar batch ekseni yerine özellik ekseni üzerindedir ($N\to d$). RMSNorm ortalama çıkarmadığından ikinci terim düşer: $dx_k=\frac1{d\,r}\big(d\,d\hat x_k-\hat x_k\sum_i d\hat x_i\hat x_i\big)$, $r=\sqrt{\frac1d\sum x_i^2+\varepsilon}$.

### Maliyet ve çıkarımda katlama {#katlama}

BN modele karmaşıklık ekler: her BN katmanı **4 × (düğüm sayısı)** parametre getirir — yarısı ($\gamma,\beta$) geri yayılımla öğrenilir, yarısı ($\mu,\sigma^2$) veriden hesaplanır. Ayrıca her katmanda ek hesap yapıldığı için tahmin yavaşlar. Neyse ki eğitimden sonra BN çoğu zaman **önceki katmana katlanabilir** (fuse): önceki katmanın ağırlıklarını ve biasını, doğrudan doğru ölçek ve kaymayı üretecek biçimde güncelleriz.

:::türetim BN'i doğrusal katmana katlamak
Önceki katman $XW+b$ hesaplasın ($W$: girdi × çıktı, $b,\gamma,\beta,\mu,\sigma^2$: çıktı boyunda). Çıkarımda $\mu,\sigma^2$ **sabittir**, dolayısıyla BN bir afin dönüşümdür:
$$\gamma\otimes\frac{XW+b-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta=X\underbrace{\Big(W\otimes\frac{\gamma}{\sqrt{\sigma^2+\varepsilon}}\Big)}_{W'}+\underbrace{\gamma\otimes\frac{b-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta}_{b'}=XW'+b'.$$
$W'$'de bölme sütun başınadır ($W$'nin $j$. çıktı sütunu $\gamma_j/\sqrt{\sigma_j^2+\varepsilon}$ ile çarpılır). Konvolüsyonda aynısı çıktı kanalı başına yapılır. Önceki katmanın $(W,b)$'si $(W',b')$ ile değiştirilir, BN katmanı silinir: çıkarım maliyeti sıfır. (Eğitim modunda katlanamaz: $\mu_B,\sigma_B$ her batch'te değişir.)
:::

:::uyarı Kaynak düzeltmesi
Ders slaytlarında katlama formülü $W'=\gamma\otimes W/(\sigma+\varepsilon)$, $b'=\gamma\otimes(b-\mu)/(\sigma+\varepsilon)+\beta$ olarak yazılmış. BN $\varepsilon$'u varyansa ekler: doğrusu $\sqrt{\sigma^2+\varepsilon}$'dur. $\varepsilon=10^{-5}$ ve $\sigma\approx1$ iken fark küçüktür, ama ölü ya da neredeyse sabit bir kanalda ($\sigma\approx0$) $\sigma+\varepsilon=10^{-5}$ ile $\sqrt{\varepsilon}\approx3.2\cdot10^{-3}$ arasında 300 kat fark vardır ve katlanmış model orijinalinden ayrışır. x2 ekinde katlamayı yazıp orijinalle karşılaştırıyorsun.
:::

BN derin ağlarda o kadar yaygınlaştı ki diyagramlarda çoğu zaman çizilmez bile: her katmandan sonra olduğu varsayılır.

### Normalizasyon ailesi: hangi eksenler? {#eksenler}

BN'in iki zayıf yanı vardır: (1) çıktı batch'e bağlıdır — küçük batch'lerde (tespit, segmentasyon, büyük modeller; GPU başına 1–2 görüntü) istatistikler gürültülü olur; (2) eğitim ve çıkarım farklı hesaplardır. Değişken uzunluklu dizilerde (RNN, Transformer) "batch ortalaması" dolgu (padding) ile kirlenir ve zaman adımları arasında anlamsızlaşır; BN'in RNN'lerde kullanımı bu yüzden zordur (RNN'lerde patlayan gradyanlara karşı genelde gradyan kırpma kullanılır, M3.7). Çözüm: istatistiği **başka eksenlerden** almak.

<figure class="fig wide"><svg viewBox="0 0 760 200" role="img" aria-label="Dört küp: BatchNorm bir kanalın tüm örnek ve konumlarını, LayerNorm bir örneğin tüm kanal ve konumlarını, InstanceNorm bir örnek ve bir kanalın konumlarını, GroupNorm bir örnek ve bir kanal grubunun konumlarını normalize eder"><polygon points="46.0,40.0 142.0,40.0 142.0,136.0 46.0,136.0" class="mute"/><polygon points="46.0,40.0 142.0,40.0 182.0,12.0 86.0,12.0" class="mute"/><polygon points="142.0,40.0 182.0,12.0 182.0,108.0 142.0,136.0" class="mute"/><line x1="62.0" y1="40.0" x2="62.0" y2="136.0" class="mute"/><line x1="62.0" y1="40.0" x2="102.0" y2="12.0" class="mute"/><line x1="78.0" y1="40.0" x2="78.0" y2="136.0" class="mute"/><line x1="78.0" y1="40.0" x2="118.0" y2="12.0" class="mute"/><line x1="94.0" y1="40.0" x2="94.0" y2="136.0" class="mute"/><line x1="94.0" y1="40.0" x2="134.0" y2="12.0" class="mute"/><line x1="110.0" y1="40.0" x2="110.0" y2="136.0" class="mute"/><line x1="110.0" y1="40.0" x2="150.0" y2="12.0" class="mute"/><line x1="126.0" y1="40.0" x2="126.0" y2="136.0" class="mute"/><line x1="126.0" y1="40.0" x2="166.0" y2="12.0" class="mute"/><line x1="56.0" y1="33.0" x2="152.0" y2="33.0" class="mute"/><line x1="152.0" y1="33.0" x2="152.0" y2="129.0" class="mute"/><line x1="66.0" y1="26.0" x2="162.0" y2="26.0" class="mute"/><line x1="162.0" y1="26.0" x2="162.0" y2="122.0" class="mute"/><line x1="76.0" y1="19.0" x2="172.0" y2="19.0" class="mute"/><line x1="172.0" y1="19.0" x2="172.0" y2="115.0" class="mute"/><polygon points="46.0,40.0 62.0,40.0 62.0,136.0 46.0,136.0" class="cp-f" fill-opacity="0.55"/><polygon points="46.0,40.0 62.0,40.0 102.0,12.0 86.0,12.0" class="cp-f" fill-opacity="0.75"/><polyline points="46.0,40.0 142.0,40.0 142.0,136.0 46.0,136.0" class="ink" fill="none"/><polyline points="46.0,40.0 86.0,12.0 182.0,12.0 182.0,108.0 142.0,136.0" class="ink" fill="none"/><polyline points="142.0,40.0 182.0,12.0" class="ink" fill="none"/><text x="114.0" y="162.0" text-anchor="middle" class="tp">BatchNorm</text><text x="114.0" y="178.0" text-anchor="middle" class="sm">kanal başına: N, H, W</text><text x="94.0" y="148.0" text-anchor="middle" class="sm">C →</text><text x="42.0" y="92.0" text-anchor="end" class="sm">H, W</text><text x="56.0" y="10.0" class="sm">N ↗</text><polygon points="226.0,40.0 322.0,40.0 322.0,136.0 226.0,136.0" class="mute"/><polygon points="226.0,40.0 322.0,40.0 362.0,12.0 266.0,12.0" class="mute"/><polygon points="322.0,40.0 362.0,12.0 362.0,108.0 322.0,136.0" class="mute"/><line x1="242.0" y1="40.0" x2="242.0" y2="136.0" class="mute"/><line x1="242.0" y1="40.0" x2="282.0" y2="12.0" class="mute"/><line x1="258.0" y1="40.0" x2="258.0" y2="136.0" class="mute"/><line x1="258.0" y1="40.0" x2="298.0" y2="12.0" class="mute"/><line x1="274.0" y1="40.0" x2="274.0" y2="136.0" class="mute"/><line x1="274.0" y1="40.0" x2="314.0" y2="12.0" class="mute"/><line x1="290.0" y1="40.0" x2="290.0" y2="136.0" class="mute"/><line x1="290.0" y1="40.0" x2="330.0" y2="12.0" class="mute"/><line x1="306.0" y1="40.0" x2="306.0" y2="136.0" class="mute"/><line x1="306.0" y1="40.0" x2="346.0" y2="12.0" class="mute"/><line x1="236.0" y1="33.0" x2="332.0" y2="33.0" class="mute"/><line x1="332.0" y1="33.0" x2="332.0" y2="129.0" class="mute"/><line x1="246.0" y1="26.0" x2="342.0" y2="26.0" class="mute"/><line x1="342.0" y1="26.0" x2="342.0" y2="122.0" class="mute"/><line x1="256.0" y1="19.0" x2="352.0" y2="19.0" class="mute"/><line x1="352.0" y1="19.0" x2="352.0" y2="115.0" class="mute"/><polygon points="226.0,40.0 322.0,40.0 322.0,136.0 226.0,136.0" class="cp-f" fill-opacity="0.55"/><polygon points="226.0,40.0 322.0,40.0 332.0,33.0 236.0,33.0" class="cp-f" fill-opacity="0.75"/><polygon points="322.0,40.0 332.0,33.0 332.0,129.0 322.0,136.0" class="cp-f" fill-opacity="0.4"/><polyline points="226.0,40.0 322.0,40.0 322.0,136.0 226.0,136.0" class="ink" fill="none"/><polyline points="226.0,40.0 266.0,12.0 362.0,12.0 362.0,108.0 322.0,136.0" class="ink" fill="none"/><polyline points="322.0,40.0 362.0,12.0" class="ink" fill="none"/><text x="294.0" y="162.0" text-anchor="middle" class="tp">LayerNorm</text><text x="294.0" y="178.0" text-anchor="middle" class="sm">örnek başına: C, H, W</text><polygon points="406.0,40.0 502.0,40.0 502.0,136.0 406.0,136.0" class="mute"/><polygon points="406.0,40.0 502.0,40.0 542.0,12.0 446.0,12.0" class="mute"/><polygon points="502.0,40.0 542.0,12.0 542.0,108.0 502.0,136.0" class="mute"/><line x1="422.0" y1="40.0" x2="422.0" y2="136.0" class="mute"/><line x1="422.0" y1="40.0" x2="462.0" y2="12.0" class="mute"/><line x1="438.0" y1="40.0" x2="438.0" y2="136.0" class="mute"/><line x1="438.0" y1="40.0" x2="478.0" y2="12.0" class="mute"/><line x1="454.0" y1="40.0" x2="454.0" y2="136.0" class="mute"/><line x1="454.0" y1="40.0" x2="494.0" y2="12.0" class="mute"/><line x1="470.0" y1="40.0" x2="470.0" y2="136.0" class="mute"/><line x1="470.0" y1="40.0" x2="510.0" y2="12.0" class="mute"/><line x1="486.0" y1="40.0" x2="486.0" y2="136.0" class="mute"/><line x1="486.0" y1="40.0" x2="526.0" y2="12.0" class="mute"/><line x1="416.0" y1="33.0" x2="512.0" y2="33.0" class="mute"/><line x1="512.0" y1="33.0" x2="512.0" y2="129.0" class="mute"/><line x1="426.0" y1="26.0" x2="522.0" y2="26.0" class="mute"/><line x1="522.0" y1="26.0" x2="522.0" y2="122.0" class="mute"/><line x1="436.0" y1="19.0" x2="532.0" y2="19.0" class="mute"/><line x1="532.0" y1="19.0" x2="532.0" y2="115.0" class="mute"/><polygon points="406.0,40.0 422.0,40.0 422.0,136.0 406.0,136.0" class="cp-f" fill-opacity="0.55"/><polygon points="406.0,40.0 422.0,40.0 432.0,33.0 416.0,33.0" class="cp-f" fill-opacity="0.75"/><polyline points="406.0,40.0 502.0,40.0 502.0,136.0 406.0,136.0" class="ink" fill="none"/><polyline points="406.0,40.0 446.0,12.0 542.0,12.0 542.0,108.0 502.0,136.0" class="ink" fill="none"/><polyline points="502.0,40.0 542.0,12.0" class="ink" fill="none"/><text x="474.0" y="162.0" text-anchor="middle" class="tp">InstanceNorm</text><text x="474.0" y="178.0" text-anchor="middle" class="sm">örnek × kanal: H, W</text><polygon points="586.0,40.0 682.0,40.0 682.0,136.0 586.0,136.0" class="mute"/><polygon points="586.0,40.0 682.0,40.0 722.0,12.0 626.0,12.0" class="mute"/><polygon points="682.0,40.0 722.0,12.0 722.0,108.0 682.0,136.0" class="mute"/><line x1="602.0" y1="40.0" x2="602.0" y2="136.0" class="mute"/><line x1="602.0" y1="40.0" x2="642.0" y2="12.0" class="mute"/><line x1="618.0" y1="40.0" x2="618.0" y2="136.0" class="mute"/><line x1="618.0" y1="40.0" x2="658.0" y2="12.0" class="mute"/><line x1="634.0" y1="40.0" x2="634.0" y2="136.0" class="mute"/><line x1="634.0" y1="40.0" x2="674.0" y2="12.0" class="mute"/><line x1="650.0" y1="40.0" x2="650.0" y2="136.0" class="mute"/><line x1="650.0" y1="40.0" x2="690.0" y2="12.0" class="mute"/><line x1="666.0" y1="40.0" x2="666.0" y2="136.0" class="mute"/><line x1="666.0" y1="40.0" x2="706.0" y2="12.0" class="mute"/><line x1="596.0" y1="33.0" x2="692.0" y2="33.0" class="mute"/><line x1="692.0" y1="33.0" x2="692.0" y2="129.0" class="mute"/><line x1="606.0" y1="26.0" x2="702.0" y2="26.0" class="mute"/><line x1="702.0" y1="26.0" x2="702.0" y2="122.0" class="mute"/><line x1="616.0" y1="19.0" x2="712.0" y2="19.0" class="mute"/><line x1="712.0" y1="19.0" x2="712.0" y2="115.0" class="mute"/><polygon points="586.0,40.0 634.0,40.0 634.0,136.0 586.0,136.0" class="cp-f" fill-opacity="0.55"/><polygon points="586.0,40.0 634.0,40.0 644.0,33.0 596.0,33.0" class="cp-f" fill-opacity="0.75"/><polyline points="586.0,40.0 682.0,40.0 682.0,136.0 586.0,136.0" class="ink" fill="none"/><polyline points="586.0,40.0 626.0,12.0 722.0,12.0 722.0,108.0 682.0,136.0" class="ink" fill="none"/><polyline points="682.0,40.0 722.0,12.0" class="ink" fill="none"/><text x="654.0" y="162.0" text-anchor="middle" class="tp">GroupNorm</text><text x="654.0" y="178.0" text-anchor="middle" class="sm">örnek × grup: C/G, H, W</text></svg><figcaption>Şekil 2 — Aynı (N, C, H·W) tensörü, dört yöntem. Renkli bölge, tek bir (μ, σ) çiftinin hesaplandığı eleman kümesidir (Wu ve He 2018, Şekil 2'nin yeniden çizimi; burada C = 6, N = 4, GroupNorm'da G = 2). Yöntemler yalnız bu kümenin nasıl seçildiğinde ayrışır; normalize et-ölçekle-kaydır adımları aynıdır.</figcaption></figure>

Bir CNN aktivasyonu $(N,C,H,W)$ shape'indedir. Tüm yöntemler aynı üç adımı yapar — bir eleman kümesi $S$ üzerinden $\mu_S,\sigma_S^2$ hesapla, $\hat x=(x-\mu_S)/\sqrt{\sigma_S^2+\varepsilon}$, ardından $\gamma\hat x+\beta$ — ve **yalnız $S$'nin seçiminde** ayrışır:

| yöntem | $S$: istatistik ekseni | (μ, σ) çifti sayısı | batch'e bağlı mı? | tipik kullanım |
|---|---|---|---|---|
| BatchNorm (Ioffe ve Szegedy 2015) | $N,H,W$ (kanal başına) | $C$ | evet; eğitim ≠ çıkarım | CNN'ler, büyük batch |
| LayerNorm (Ba ve ark. 2016) | $C,H,W$ (örnek başına) | $N$ | hayır | Transformer, RNN |
| InstanceNorm (Ulyanov ve ark. 2016) | $H,W$ (örnek × kanal) | $N\cdot C$ | hayır | stil aktarımı, üretici ağlar |
| GroupNorm (Wu ve He 2018) | $C/G,H,W$ (örnek × grup) | $N\cdot G$ | hayır | küçük batch'li tespit/segmentasyon |
| RMSNorm (Zhang ve Sennrich 2019) | LN ile aynı, ortalama çıkarılmaz | $N$ | hayır | LLaMA vb. büyük dil modelleri |

**Ara durumlar:** GroupNorm $G=1$'de LayerNorm'a, $G=C$'de InstanceNorm'a eşittir. Wu ve He varsayılan olarak $G=32$ kullandı; ResNet-50 ImageNet'te batch boyu 2'de GN'nin hatası BN'inkinden %10.6 daha düşük, büyük batch'te ise BN'e yakındı.

**InstanceNorm** her görüntünün her kanalını ayrı standartlaştırır: kontrast ve "stil" bilgisini (kanal başına ortalama ve varyans) siler. Stil aktarımında istenen tam budur; sınıflandırmada ise o bilgi gerekebilir. PyTorch'ta `InstanceNorm2d` varsayılan olarak **afin parametresizdir** (`affine=False`).

:::tanım LayerNorm ve RMSNorm (bir vektör için)
$x\in\mathbb R^d$ (bir token'ın ya da örneğin özellikleri):
$$\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta,\quad \mu=\frac1d\sum_i x_i,\ \sigma^2=\frac1d\sum_i(x_i-\mu)^2;$$
$$\mathrm{RMSNorm}(x)=g\odot\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\varepsilon}} .$$
RMSNorm ortalamayı çıkarmaz ve $\beta$ içermez: bir indirgeme (reduction) ve bir parametre vektörü daha az. Zhang ve Sennrich, LN'in başarısında asıl payın yeniden **ölçekleme** değişmezliğinde olduğunu, yeniden merkezlemenin çoğu zaman gereksiz olduğunu savundu. PyTorch'ta `nn.LayerNorm(d)` ($\gamma,\beta$ eleman başına), `nn.RMSNorm(d)` (yalnız $g$; 2.4 sürümünden beri).
:::

LN ve RMSNorm her örneği **kendi** özellikleriyle normalize eder: batch boyundan, dizi uzunluğundan ve train/eval modundan bağımsızdır; tek örnekte ve dolgulu dizilerde aynı biçimde çalışır. Aşağıdaki şekilde gerçek bir tensörde hangi hücrelerin aynı (μ, σ)'yı paylaştığını seç ve normalize sonrası neyin silindiğine bak: BN kanal ortalamalarını sıfırlar ama örnekler arası ölçek farkını korur; LN örnek ölçeklerini eşitler ama kanallar arası farkı korur.

::lab l3

### Transformer'da yer: post-norm ve pre-norm {#prepost}

<figure class="fig"><svg viewBox="0 0 420 222" role="img" aria-label="Post-norm blokta normalizasyon artık toplamadan sonra, pre-norm blokta alt katmandan önce ve artık yolunun dışında"><defs><marker id="pn1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="pn2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cq-f"/></marker></defs><text x="90" y="18" text-anchor="middle" class="tp">post-norm (2017 Transformer)</text><text x="90" y="40" text-anchor="middle">x</text><rect x="40" y="52" width="100" height="32" rx="6" class="ca"/><text x="90.0" y="72.0" text-anchor="middle">F (dikkat / MLP)</text><circle cx="90" cy="117" r="11" class="ink"/><text x="90" y="122" text-anchor="middle">+</text><rect x="40" y="150" width="100" height="32" rx="6" class="cp"/><text x="90.0" y="170.0" text-anchor="middle">LN</text><line x1="90" y1="44" x2="90" y2="51" class="ink" marker-end="url(#pn1)"/><line x1="90" y1="84" x2="90" y2="105" class="ink" marker-end="url(#pn1)"/><line x1="90" y1="128" x2="90" y2="149" class="ink" marker-end="url(#pn1)"/><line x1="90" y1="182" x2="90" y2="200" class="ink" marker-end="url(#pn1)"/><text x="90" y="214" text-anchor="middle">çıktı</text><path d="M 90 46 H 30 V 117 H 77" class="cq dash" fill="none" marker-end="url(#pn2)"/><text x="300" y="18" text-anchor="middle" class="tp">pre-norm (GPT-2, LLaMA…)</text><text x="300" y="40" text-anchor="middle">x</text><rect x="250" y="52" width="100" height="32" rx="6" class="cp"/><text x="300.0" y="72.0" text-anchor="middle">LN / RMSNorm</text><rect x="250" y="106" width="100" height="32" rx="6" class="ca"/><text x="300.0" y="126.0" text-anchor="middle">F (dikkat / MLP)</text><circle cx="300" cy="171" r="11" class="ink"/><text x="300" y="176" text-anchor="middle">+</text><line x1="300" y1="44" x2="300" y2="51" class="ink" marker-end="url(#pn1)"/><line x1="300" y1="84" x2="300" y2="105" class="ink" marker-end="url(#pn1)"/><line x1="300" y1="138" x2="300" y2="159" class="ink" marker-end="url(#pn1)"/><line x1="300" y1="182" x2="300" y2="200" class="ink" marker-end="url(#pn1)"/><text x="300" y="214" text-anchor="middle">çıktı</text><path d="M 300 46 H 240 V 171 H 287" class="cq dash" fill="none" marker-end="url(#pn2)"/></svg><figcaption>Şekil 3 — Transformer bloğunda normalizasyonun yeri. Post-norm: LN(x + F(x)) — artık yol (kesikli) her blokta normalize edilir. Pre-norm: x + F(LN(x)) — artık yol hiç normalize edilmeden en sona kadar akar; derin modellerde daha kararlıdır ve ısınmaya daha az bağımlıdır (Xiong ve ark. 2020). Pre-norm modellerde en sona genelde bir LN daha eklenir.</figcaption></figure>

Orijinal Transformer (2017) normalizasyonu artık toplamadan **sonra** koydu: $\mathrm{LN}(x+F(x))$. GPT-2'den beri yaygın olan **pre-norm** ise alt katmanın girdisini normalize eder: $x+F(\mathrm{LN}(x))$. Pre-norm'da artık yol (kimlik) hiç normalize edilmeden ağın sonuna kadar akar — gradyanlar da onun üzerinden doğrudan ilk bloklara ulaşır. Xiong ve ark. (2020), post-norm'da çıkışa yakın katmanların gradyanlarının başlangıçta büyük olduğunu ve bu yüzden öğrenme oranı **ısınmasının** (M3.4) zorunlu hâle geldiğini; pre-norm'un ısınmasız da eğitilebildiğini gösterdi. Bedeli: artık akım derinlikle büyür, bu yüzden son bloktan sonra bir normalizasyon daha eklenir. Bugünkü büyük dil modellerinin çoğu (LLaMA ailesi gibi) **pre-norm + RMSNorm** kullanır (M4.7).

### Normalizasyonsuz derin ağlar {#fixup}

Normalizasyon katmanları gerçekten gerekli mi? Zhang, Dauphin ve Ma (2019) **Fixup** başlatmasıyla 10 000 katmanlı bir artık ağı hiç BN kullanmadan eğitti ve görüntü sınıflandırmada BN'li ağlarla yarışan sonuçlar aldı. Fikir: BN'in başlangıçtaki asıl katkısı, artık dalların (residual branch) çıktısını küçük tutup ağı başta neredeyse kimlik fonksiyonu yapmaktır; bu doğrudan başlatmayla da yapılabilir.

:::ayrıntı Fixup'ın kuralları
1. Sınıflandırma katmanını ve her artık dalın **son** katmanını sıfırla başlat (başta her blok kimlik).
2. Diğer katmanları standart (He) başlat, sonra her artık dalın içindeki ağırlık katmanlarını $L^{-1/(2m-2)}$ ile ölçekle ($L$: artık dal sayısı, $m$: dal başına katman sayısı). Böylece toplam güncelleme büyüklüğü derinlikten bağımsız kalır.
3. Her dalın sonuna 1'den başlayan skaler bir çarpan, her konvolüsyon/doğrusal katmandan ve aktivasyondan önce 0'dan başlayan skaler bir bias ekle.
:::

Sonraki çalışmalar da (SkipInit, De ve Smith 2020; NFNets, Brock ve ark. 2021 — uyarlamalı gradyan kırpmayla) normalizasyonsuz ağların mümkün olduğunu gösterdi. Yine de pratikte normalizasyon katmanları varsayılan seçenek olarak kaldı: ayarlaması kolaydır ve öğrenme oranına karşı ağı sağlamlaştırır. Normalizasyonu çıkarmak istiyorsan bu yöntemlerin dikkatli başlatma ve kırpma tariflerini birlikte uygulaman gerekir.

### Pratik özet {#pratik}

- **CNN, makul batch (≥ 16–32):** `Conv(bias=False) → BatchNorm2d → ReLU`; çıkarımda `model.eval()`; dağıtımda BN'i katla.
- **Küçük batch / batch'e bağımsızlık gerekiyorsa:** GroupNorm (G = 32 iyi bir başlangıç); çok GPU'da BN istiyorsan `SyncBatchNorm` ile istatistikleri cihazlar arasında topla.
- **Transformer / RNN / değişken uzunluklu dizi:** LayerNorm ya da RMSNorm, pre-norm yerleşimi.
- **BN + ağırlık çürümesi:** BN'den önceki ağırlıklarda weight decay etkin öğrenme oranını değiştirir; $\gamma,\beta$ ve biaslar genelde weight decay'den muaf tutulur.
- **Teşhis:** eğitim ve değerlendirme doğruluğu arasında beklenmedik büyük fark → önce `train()/eval()` modlarını ve koşu istatistiklerini kontrol et.

### Özet {#ozet}

- BN her özelliği batch istatistikleriyle standartlaştırır, sonra öğrenilen $\gamma,\beta$ ile yeniden ölçekler; eğitimde batch'e, çıkarımda sabit koşu istatistiklerine dayanan **iki farklı fonksiyondur**.
- Etkileri: daha büyük öğrenme oranı, başlatmaya daha az duyarlılık, düzenlileştirme; ölçek değişmezliği ($\nabla L(cW)=\frac1c\nabla L(W)$, gradyan $\perp W$) bunun açıklamalarından biridir.
- Geri geçiş, $\mu$ ve $\sigma$ üzerinden geçen dolaylı yolları da toplar: $dx=\frac1{N\sigma}(N\,d\hat x-\sum d\hat x-\hat x\sum d\hat x\hat x)$.
- Çıkarımda BN önceki katmana katlanır: $W'=W\otimes\gamma/\sqrt{\sigma^2+\varepsilon}$.
- BN/LN/IN/GN/RMSNorm yalnız istatistiğin alındığı eksenlerde farklıdır; batch'ten bağımsız olanlar (LN, GN, RMSNorm) küçük batch'lerde ve dizilerde tercih edilir.
- Artık bağlantısız derin BN ağlarında başlangıçta gradyanlar ilk katmanlara doğru patlar; normalizasyon artık bağlantılarla birlikte düşünülmelidir.
