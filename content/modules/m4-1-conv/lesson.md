### Neden konvolüsyon? Görsel korteksten CNN'e {#neden}

Konvolüsyonel sinir ağları (CNN) beynin **görsel korteksinin** incelenmesinden doğdu ve 1980'lerden beri görüntü tanımada kullanılıyor; bugün görüntü arama, otonom sürüş ve video sınıflandırmanın yanı sıra ses tanıma ve dil işlemede de başarılılar. (Belirli görevlerde Vision Transformer mimarileri en iyi CNN'lere yetişiyor ya da onları geçiyor, M4.15.)

**Biyolojik esin.** Beynin dış katmanı olan serebral korteks (kalınlığı 2–4 mm, ama beyin kütlesinin yaklaşık yarısı; 14–16 milyar nöron) büyük ölçüde altı katmanlı neokorteksten oluşur. D. H. Hubel ve T. Wiesel 1958–59'da kediler (birkaç yıl sonra maymunlar) üzerindeki deneyleriyle görsel korteksin yapısına dair kritik bulgular elde ettiler (1981 Nobel Fizyoloji/Tıp Ödülü):

- Görsel korteksteki birçok nöronun küçük, **yerel bir alıcı alanı** (receptive field) vardır: yalnız görme alanının sınırlı bir bölgesindeki uyaranlara tepki verir. Farklı nöronların alıcı alanları örtüşebilir ve birlikte tüm görme alanını döşer.
- Bazı nöronlar yalnız yatay çizgilere, diğerleri farklı yönelimli çizgilere tepki verir; aynı alıcı alana sahip iki nöron farklı yönelimlere duyarlı olabilir.
- Bazı nöronların alıcı alanları daha büyüktür ve alt düzey desenlerin **birleşimi** olan daha karmaşık desenlere tepki verir.

Buradan, üst düzey nöronların komşu alt düzey nöronların çıktılarına dayandığı fikri doğdu: görme alanının herhangi bir yerindeki her türlü karmaşık deseni tespit edebilen güçlü bir mimari. (Bu "hiyerarşik aşağıdan yukarı duyusal model" bugün birçok deneysel sonucu açıklayamadığı için yoğun biçimde sorgulanıyor; CNN'ler için bir esin kaynağı, beynin doğru bir modeli değil.) Bu çalışmalar 1980'de Fukushima'nın **neokognitron**'una ilham verdi; o da zamanla bugünkü CNN'lere evrildi. Önemli bir dönüm noktası LeCun ve ark.'nın 1998 makalesi "Gradient-Based Learning Applied to Document Recognition"dır: bankalarda çek üzerindeki el yazısı rakamları okumak için yaygın kullanılan **LeNet-5**'i tanıttı (M4.2). LeNet-5 tam bağlı katmanlar ve sigmoid gibi bildiğimiz yapı taşlarının yanında iki yeni yapı taşı getirdi: **konvolüsyon katmanları** ve **havuzlama katmanları**.

**Tam bağlı katmanlar neden yetmiyor?** Küçük görüntülerde (MNIST) çalışırlar, ama büyük görüntülerde gereken parametre sayısı patlar. 100 × 100'lük bir görüntüde 10 000 piksel var; ilk katman yalnız 1 000 nöronlu olsa bile (ki bu bir sonraki katmana aktarılan bilgiyi zaten ciddi biçimde kısıtlar) **10 milyon bağlantı** eder — ve bu yalnız ilk katman. Normalde gizli katmanın girdiden **daha geniş** olmasını isteriz. CNN'ler bu sorunu **kısmi bağlantı** ve **ağırlık paylaşımıyla** çözer.

:::sezgi CNN'ler neden doğal görüntülerde bu kadar iyi çalışır?
Görüntüleri bir piksel ızgarası üzerinde tanımlı rastgele bir süreç olarak modellersek, doğal ve anlamlı görüntülerde bu sürecin özilinti (autocorrelation) fonksiyonu piksel kaymasıyla hızla söner ve süreç en azından geniş anlamda durağandır (wide-sense stationary). Somut olarak:
1. Yerel görsel yapıları tanımlayan piksel–piksel ilişkileri **küçük yamalarla** sınırlıdır → her katmandaki filtreler görüntüden çok küçük olabilir.
2. Bu ilişkiler ve tanımladıkları yapılar görüntüdeki **konumdan bağımsızdır** → aynı filtre tüm görüntüde kullanılabilir (ağırlık paylaşımı).
3. Büyük ölçekli öznitelikler (uzun kenarlar, az eğrilikli uzun eğriler) CNN'in **hiyerarşik yapısıyla** yakalanır → katmanlar üst üste kondukça alıcı alan büyür.
:::

Görüntü işleme ve bilgisayarlı görüde **çok ölçekli** (görüntü piramidi) işlem de yaygındır: bazen yalnız hesap içindir (hareket kestiriminde en kaba ölçekte yerel bir arama penceresiyle başlayıp ince ölçeklerde iyileştirmek), bazen düzenlileştirici gibi sağlamlık katar, bazen de nesnenin hareketle değişen boyutu ölçeğe bağlı işlem gerektirir. CNN'in katmandan katmana küçülen ama derinleşen yapısı bu fikrin öğrenilen versiyonudur.

### Konvolüsyon işlemi {#islem}

Konvolüsyon sinyal işlemenin temel işlemidir; 2-B konvolüsyon görüntülerin bazı özelliklerini güçlendirmek, bastırmak ya da seçici biçimde değiştirmek için kullanılır.

:::tanım 2-B konvolüsyon ve çapraz korelasyon
$x$ girdi görüntüsü, $w$ 2-B filtre (çekirdek), $y$ çıktı:
$$y[i,j]=(w*x)[i,j]=\sum_{m}\sum_{n}w[m,n]\;x[i-m,\,j-n]=\sum_m\sum_n w[i-m,\,j-n]\;x[m,n].$$
**Çapraz korelasyon** ($\star$) çekirdeği çevirmeden kaydırır:
$$(w\star x)[i,j]=\sum_m\sum_n w[m,n]\;x[i+m,\,j+n].$$
İkisi arasındaki tek fark çekirdeğin 180° döndürülmesidir: $w*x=\mathrm{rot}_{180}(w)\star x$.
:::

Formül ilk bakışta zorlayıcı görünebilir; uygulaması basittir:
1. Filtreyi bir kez dikey, bir kez yatay çevir (merkezi etrafında 180° döndür).
2. Filtrenin merkezini girdideki bir pikselin üstüne getir.
3. Filtre katsayılarını örtüşen piksel değerleriyle çarp ve topla.
4. Toplamı çıktıya yaz; sonraki piksele geç.

:::uyarı Derin öğrenmede "konvolüsyon" aslında korelasyondur
PyTorch'un `nn.Conv2d`'si, TensorFlow ve neredeyse tüm derin öğrenme kütüphaneleri çekirdeği **çevirmeden** uygular: matematiksel olarak çapraz korelasyon hesaplarlar. Ağırlıklar öğrenildiği için bu fark önemsizdir — ağ çevrilmiş çekirdeği öğrenir. Ama elle tasarlanmış bir filtreyi (Sobel, asimetrik bir çekirdek) kütüphane konvolüsyonuyla uygularken ya da bir sinyal işleme kitabındaki formülü kodlarken işaret ve yön farkına dikkat et.
:::

**Kenarlar ve dolgu.** 1×1'den büyük her çekirdekte kenarlardaki pikseller için bazı girdi değerleri eksiktir. İki seçenek var: tüm komşuları olmayan pikselleri **atlamak** (çıktı küçülür; "valid") ya da girdiyi **dolgulamak** (pad) ki kenar piksellerinin de gereken komşuları olsun. Dolgu yöntemleri: (1) **sıfır dolgu** (en yaygın; kenarda yapay bir karanlık çerçeve ekler), (2) **kenar pikselini tekrarlama** (replicate), (3) **aynalama** (reflect). PyTorch'ta `nn.Conv2d(..., padding=1, padding_mode="zeros" | "replicate" | "reflect" | "circular")`.

**Ne işe yarar?** Bulanıklaştırma, keskinleştirme, gürültü azaltma, şekil/desen tespiti… Ders slaytlarındaki klasik 3×3 filtreler:

| filtre | çekirdek | etkisi |
|---|---|---|
| kutu bulanıklığı | $\frac19\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}$ | komşuluğun ortalaması; gürültüyü ve ayrıntıyı siler |
| Gauss | $\frac1{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}$ | merkeze ağırlık veren daha yumuşak bulanıklık |
| keskinleştirme | $\frac12\begin{bmatrix}-1&-1&-1\\-1&10&-1\\-1&-1&-1\end{bmatrix}$ | katsayıların toplamı 1: düz bölgeler korunur, komşulardan farklı pikseller vurgulanır |
| yatay kenar (Sobel) | $\begin{bmatrix}1&2&1\\0&0&0\\-1&-2&-1\end{bmatrix}$ | dikey yönde parlaklık değişimi → yatay kenarlar |
| dikey kenar (Sobel) | $\begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix}$ | yatay yönde parlaklık değişimi → dikey kenarlar |

Katsayılarının toplamı 1 olan filtreler (bulanıklık, keskinleştirme) sabit bölgelerin parlaklığını korur; toplamı 0 olanlar (Sobel) sabit bölgelerde 0 verir ve yalnız **değişime** tepki gösterir.

Aşağıdaki deneyde bir girdi ve çekirdek seç, bir çıktı hücresine tıkla: alıcı alanı çerçevelenir ve toplam terim terim yazılır. "Kaydırma" çekirdeği ve "tek nokta" girdisiyle korelasyon–konvolüsyon farkını gör: korelasyon görüntüyü sola, konvolüsyon sağa kaydırır; dürtünün konvolüsyon yanıtı çekirdeğin kendisidir.

::lab l2

:::türetim Kaymaya eşdeğişirlik (translation equivariance)
Girdiyi $s$ kadar kaydır: $x'[n]=x[n-s]$ (1-B, adım 1, kenar etkilerini yok say). Korelasyonun çıktısı:
$$y'[i]=\sum_u w[u]\,x'[i+u]=\sum_u w[u]\,x[i-s+u]=y[i-s].$$
Yani **önce kaydırıp sonra konvolüsyon** ile **önce konvolüsyon sonra kaydırma** aynıdır: $\mathrm{conv}(T_sx)=T_s\,\mathrm{conv}(x)$. Bir deseni bir yerde tanıyan filtre onu her yerde tanır — ağırlık paylaşımının matematiksel karşılığı. **Koşullar:** adım 1 (adım $S$ ise yalnız $S$'nin katı kaymalarda), kenarlardan uzakta (dolgu kenarda eşitliği bozar). Değişmezlik (invariance, çıktının hiç değişmemesi) ayrı bir özelliktir; ağın sonundaki havuzlama ve global ortalama ile kısmen elde edilir.
:::

### Konvolüsyon katmanı {#katman}

CNN'in en önemli yapı taşı **konvolüsyon katmanıdır**. Bir konvolüsyon katmanının bir ya da daha çok **konvolüsyon nöronu** vardır ve bunlar şimdiye kadarki tam bağlı nöronlardan farklıdır:

- Bir konvolüsyon nöronu, önceki katmandan gelen aktivasyonların **küçük bir yamasına** bağlanıp tek bir çıktı aktivasyonu üretir. Bu yamaya **alıcı alan** denir; konvolüsyon işlemindeki filtreye karşılık gelir.
- Konvolüsyon işlemi tüm girdi konumlarını ziyaret ederek önceki katmanın tüm aktivasyonlarını "görür". Her konumda bir çıktı ürettiği için sonuç, girdiye benzer ya da (dolguya bağlı olarak) ona eşit boyutludur.
- İlk konvolüsyon katmanındaki bir nöron, girdi görüntüsünün her pikseline değil yalnız alıcı alanındakilere bağlıdır. İkinci katmandaki her nöron da birinci katmanın yalnız küçük bir dikdörtgenine bağlıdır. Ağ ilk gizli katmanda küçük, alt düzey özniteliklere odaklanır, sonraki katmanlarda bunları daha büyük, üst düzey özniteliklerde birleştirir — gerçek dünya görüntülerinde bu hiyerarşi yaygındır.

Tam bağlı ağlarda görüntüyü 1-B'ye düzleştirmemiz gerekiyordu; CNN'de her katman **2-B** temsil edilir, bu da nöronları karşılık gelen girdileriyle eşleştirmeyi kolaylaştırır.

**Konumlar.** Bir katmanın $i$. satır, $j$. sütunundaki aktivasyon, önceki katmanın $i\ldots i+f_h-1$ satırlarındaki ve $j\ldots j+f_w-1$ sütunlarındaki aktivasyonlardan hesaplanır ($f_h,f_w$: alıcı alanın yüksekliği ve genişliği). Bir katmanın öncekiyle aynı yükseklik ve genişlikte olması için girdinin çevresine sıfırlar eklemek yaygındır (sıfır dolgu; alternatifleri kenar tekrarı ve aynalama). Alıcı alanlar aralıklı yerleştirilerek büyük bir girdi katmanı çok daha küçük bir katmana da bağlanabilir; bu, modelin hesap karmaşıklığını büyük ölçüde azaltır. Bir alıcı alandan sonrakine kaymaya **adım** (stride) denir: üst katmandaki $(i,j)$ aktivasyonu önceki katmanın $i\,s_h\ldots i\,s_h+f_h-1$ satırlarından ve $j\,s_w\ldots j\,s_w+f_w-1$ sütunlarından hesaplanır; adım iki yönde farklı olabilir.

<figure class="fig wide"><svg viewBox="0 0 420 210" role="img" aria-label="Beş satır yedi sütunluk girdi, bir hücrelik sıfır dolguyla; üç çarpı üç pencereler ikişer hücre kayarak üç satır dört sütunluk çıktıyı üretir"><rect x="20" y="34" width="22" height="22" class="mute dash"/><rect x="42" y="34" width="22" height="22" class="mute dash"/><rect x="64" y="34" width="22" height="22" class="mute dash"/><rect x="86" y="34" width="22" height="22" class="mute dash"/><rect x="108" y="34" width="22" height="22" class="mute dash"/><rect x="130" y="34" width="22" height="22" class="mute dash"/><rect x="152" y="34" width="22" height="22" class="mute dash"/><rect x="174" y="34" width="22" height="22" class="mute dash"/><rect x="196" y="34" width="22" height="22" class="mute dash"/><rect x="20" y="56" width="22" height="22" class="mute dash"/><rect x="42" y="56" width="22" height="22" class="mute"/><rect x="64" y="56" width="22" height="22" class="mute"/><rect x="86" y="56" width="22" height="22" class="mute"/><rect x="108" y="56" width="22" height="22" class="mute"/><rect x="130" y="56" width="22" height="22" class="mute"/><rect x="152" y="56" width="22" height="22" class="mute"/><rect x="174" y="56" width="22" height="22" class="mute"/><rect x="196" y="56" width="22" height="22" class="mute dash"/><rect x="20" y="78" width="22" height="22" class="mute dash"/><rect x="42" y="78" width="22" height="22" class="mute"/><rect x="64" y="78" width="22" height="22" class="mute"/><rect x="86" y="78" width="22" height="22" class="mute"/><rect x="108" y="78" width="22" height="22" class="mute"/><rect x="130" y="78" width="22" height="22" class="mute"/><rect x="152" y="78" width="22" height="22" class="mute"/><rect x="174" y="78" width="22" height="22" class="mute"/><rect x="196" y="78" width="22" height="22" class="mute dash"/><rect x="20" y="100" width="22" height="22" class="mute dash"/><rect x="42" y="100" width="22" height="22" class="mute"/><rect x="64" y="100" width="22" height="22" class="mute"/><rect x="86" y="100" width="22" height="22" class="mute"/><rect x="108" y="100" width="22" height="22" class="mute"/><rect x="130" y="100" width="22" height="22" class="mute"/><rect x="152" y="100" width="22" height="22" class="mute"/><rect x="174" y="100" width="22" height="22" class="mute"/><rect x="196" y="100" width="22" height="22" class="mute dash"/><rect x="20" y="122" width="22" height="22" class="mute dash"/><rect x="42" y="122" width="22" height="22" class="mute"/><rect x="64" y="122" width="22" height="22" class="mute"/><rect x="86" y="122" width="22" height="22" class="mute"/><rect x="108" y="122" width="22" height="22" class="mute"/><rect x="130" y="122" width="22" height="22" class="mute"/><rect x="152" y="122" width="22" height="22" class="mute"/><rect x="174" y="122" width="22" height="22" class="mute"/><rect x="196" y="122" width="22" height="22" class="mute dash"/><rect x="20" y="144" width="22" height="22" class="mute dash"/><rect x="42" y="144" width="22" height="22" class="mute"/><rect x="64" y="144" width="22" height="22" class="mute"/><rect x="86" y="144" width="22" height="22" class="mute"/><rect x="108" y="144" width="22" height="22" class="mute"/><rect x="130" y="144" width="22" height="22" class="mute"/><rect x="152" y="144" width="22" height="22" class="mute"/><rect x="174" y="144" width="22" height="22" class="mute"/><rect x="196" y="144" width="22" height="22" class="mute dash"/><rect x="20" y="166" width="22" height="22" class="mute dash"/><rect x="42" y="166" width="22" height="22" class="mute dash"/><rect x="64" y="166" width="22" height="22" class="mute dash"/><rect x="86" y="166" width="22" height="22" class="mute dash"/><rect x="108" y="166" width="22" height="22" class="mute dash"/><rect x="130" y="166" width="22" height="22" class="mute dash"/><rect x="152" y="166" width="22" height="22" class="mute dash"/><rect x="174" y="166" width="22" height="22" class="mute dash"/><rect x="196" y="166" width="22" height="22" class="mute dash"/><rect x="20" y="34" width="66" height="66" class="ca w3" fill="none"/><rect x="64" y="34" width="66" height="66" class="cp w3" fill="none"/><rect x="20" y="78" width="66" height="66" class="cb w3" fill="none"/><text x="20" y="22" class="sm">girdi 5 × 7 (+ kesikli: sıfır dolgu P = 1)</text><rect x="290" y="78" width="22" height="22" class="soft-a ca"/><rect x="312" y="78" width="22" height="22" class="soft-p cp"/><rect x="334" y="78" width="22" height="22" class="mute"/><rect x="356" y="78" width="22" height="22" class="mute"/><rect x="290" y="100" width="22" height="22" class="soft-b cb"/><rect x="312" y="100" width="22" height="22" class="mute"/><rect x="334" y="100" width="22" height="22" class="mute"/><rect x="356" y="100" width="22" height="22" class="mute"/><rect x="290" y="122" width="22" height="22" class="mute"/><rect x="312" y="122" width="22" height="22" class="mute"/><rect x="334" y="122" width="22" height="22" class="mute"/><rect x="356" y="122" width="22" height="22" class="mute"/><text x="290" y="68" class="sm">çıktı 3 × 4</text><text x="290" y="164" class="sm">⌊(5 + 2 − 3)/2⌋ + 1 = 3</text><text x="290" y="180" class="sm">⌊(7 + 2 − 3)/2⌋ + 1 = 4</text><text x="20" y="206" class="sm">komşu pencereler 2 hücre kayar (adım 2)</text></svg><figcaption>Şekil 1 — Adımlı konvolüsyon. Çıktının (i, j) hücresi, girdinin (dolgulu koordinatlarda) i·s<sub>h</sub> … i·s<sub>h</sub> + f<sub>h</sub> − 1 satırlarından ve j·s<sub>w</sub> … j·s<sub>w</sub> + f<sub>w</sub> − 1 sütunlarından hesaplanır. Renkli çerçeveler üç çıktı hücresinin alıcı alanlarıdır; komşu alanlar örtüşür (Géron, Bölüm 14'teki örnek).</figcaption></figure>

:::türetim Çıktı boyu
Girdi uzunluğu $H$, her iki yandan $P$ dolgu, çekirdek $K$, adım $S$. Dolgulu uzunluk $H+2P$; ilk pencere 0. konumda başlar, son geçerli başlangıç konumu $H+2P-K$'dır (pencere taşmamalı). Başlangıçlar $0,S,2S,\dots$ olduğundan sayıları
$$H_{\text{out}}=\Big\lfloor\frac{H+2P-K}{S}\Big\rfloor+1 .$$
**Genişleme** (dilation) $D$ ile çekirdeğin elemanları arasında $D-1$ boşluk bırakılır, etkin boyu $D(K-1)+1$ olur: formülde $K\to D(K-1)+1$. "Aynı" (same) dolgu ($S=1$'de $H_{\text{out}}=H$) tek $K$ için $P=(K-1)/2$'dir; PyTorch'ta `padding="same"` (yalnız adım 1'de).
:::

**Filtreler ve öznitelik haritaları.** Bir konvolüsyon nöronunun ağırlıkları alıcı alan boyutunda küçük bir görüntü gibi gösterilebilir: **filtre** (konvolüsyon çekirdeği). Örneğin ortası beyaz dikey bir çizgi, geri kalanı siyah olan 7×7 bir filtre (orta sütun 1, gerisi 0): bu ağırlıkları kullanan nöron alıcı alanındaki orta dikey çizgi dışında her şeyi yok sayar; çıktıda dikey beyaz çizgiler güçlenir, gerisi bulanıklaşır. Ortası yatay çizgi olan filtre de yatay çizgileri güçlendirir. Her konvolüsyon nöronunun her çıktı konumunda toplama eklenen kendi **bias** terimi de vardır (öznitelik haritasının genel parlaklığını ayarlayan bir düğme gibi). Aynı nöronun (aynı ağırlıklar ve bias) tüm konumlarda hesapladığı aktivasyonlardan oluşan katmana **öznitelik haritası** (feature map) denir: görüntünün filtreyi en çok etkinleştiren bölgelerini vurgular.

**Çok sayıda harita.** Gerçekte bir konvolüsyon katmanında her biri kendi ağırlık ve bias'ı olan birçok konvolüsyon nöronu vardır; her biri bir öznitelik haritası üretir, dolayısıyla çıktı 3-B'dir. Bir nöronun alıcı alanı yine aynı tanımdır ama önceki katmanın **tüm** öznitelik haritalarına uzanır: konvolüsyon, filtrenin derinliği girdi derinliğine eşitlenerek 3-B girdiye genişletilir (girdi 2-B → derinlik 1 → filtre $w\times h\times1$; girdi 3-B, derinlik $k$ → filtre $w\times h\times k$). Kısacası bir konvolüsyon katmanı girdisine aynı anda birden çok eğitilebilir filtre uygular ve girdinin **her yerinde** birden çok özniteliği tespit edebilir. Girdi görüntüleri de birden çok alt katmandan oluşur: renk kanalı başına bir (genelde RGB: 3; gri tonlamalı: 1; kızılötesi gibi ek frekansları yakalayan hiperspektral uydu görüntüleri: yüzlerce).

<figure class="fig"><svg viewBox="0 0 440 225" role="img" aria-label="Girdi tensörü C kanallı; her biri C kanal derinliğinde F filtre; çıktı F kanallı"><defs><marker id="cm1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><polygon points="50,30 140,30 110,60 20,60" class="soft-a ca"/><polygon points="110,60 140,30 140,120 110,150" class="soft-a ca"/><rect x="20" y="60" width="90" height="90" class="soft-a ca"/><text x="65" y="166" text-anchor="middle" class="sm">girdi C × H × W</text><rect x="44" y="84" width="26" height="26" class="cp w2" fill="none"/><polygon points="202,28 228,28 216,40 190,40" class="soft-p cp"/><polygon points="216,40 228,28 228,54 216,66" class="soft-p cp"/><rect x="190" y="40" width="26" height="26" class="soft-p cp"/><text x="203" y="82" text-anchor="middle" class="sm"></text><polygon points="202,88 228,88 216,100 190,100" class="soft-p cp"/><polygon points="216,100 228,88 228,114 216,126" class="soft-p cp"/><rect x="190" y="100" width="26" height="26" class="soft-p cp"/><text x="203" y="142" text-anchor="middle" class="sm"></text><polygon points="202,148 228,148 216,160 190,160" class="soft-b cb"/><polygon points="216,160 228,148 228,174 216,186" class="soft-b cb"/><rect x="190" y="160" width="26" height="26" class="soft-b cb"/><text x="203" y="202" text-anchor="middle" class="sm"></text><text x="214" y="215" text-anchor="middle" class="sm">F filtre, her biri C × K × K</text><text x="214" y="138" text-anchor="middle" class="sm">⋮</text><line x1="140" y1="105" x2="182" y2="105" class="ink" marker-end="url(#cm1)"/><line x1="244" y1="105" x2="286" y2="105" class="ink" marker-end="url(#cm1)"/><polygon points="340,26 420,26 380,66 300,66" class="soft-p cp"/><polygon points="380,66 420,26 420,106 380,146" class="soft-p cp"/><rect x="300" y="66" width="80" height="80" class="soft-p cp"/><text x="340" y="162" text-anchor="middle" class="sm">çıktı F × H′ × W′</text><text x="300" y="30" class="sm">her filtre bir öznitelik haritası</text></svg><figcaption>Şekil 2 — Çok kanallı konvolüsyon katmanı. Her filtre girdinin <b>tüm</b> kanallarına uzanır (derinliği C); bir konumda C·K·K çarpımın toplamı + bias tek bir çıktı sayısıdır. F filtre F öznitelik haritası üretir. Aynı (i, j) konumundaki farklı haritaların hücreleri girdinin aynı alıcı alanına bağlıdır. Ağırlık tensörü PyTorch'ta (F, C, K, K) shape'indedir.</figcaption></figure>

:::tanım Konvolüsyon katmanının ağırlıklı toplamı
$l$. katmandaki $k$. öznitelik haritasının $(i,j)$ konumunda, aktivasyon fonksiyonuna giren ağırlıklı toplam:
$$z_{i,j,k}=b_k+\sum_{u=0}^{f_h-1}\sum_{v=0}^{f_w-1}\sum_{k'=0}^{f_{n'}-1}x_{i',j',k'}\;w_{u,v,k',k},\qquad i'=i\,s_h+u,\quad j'=j\,s_w+v .$$
- $s_h,s_w$: dikey ve yatay adım; $f_h,f_w$: alıcı alanın yüksekliği ve genişliği; $f_{n'}$: önceki katmandaki ($l-1$) öznitelik haritası sayısı.
- $x_{i',j',k'}$: $l-1$. katmanın $i'$. satır, $j'$. sütun, $k'$. haritasındaki (önceki katman girdi katmanıysa $k'$. kanalındaki) değer.
- $b_k$: $k$. haritanın bias'ı; $w_{u,v,k',k}$: $k$. haritadaki herhangi bir nöronla, alıcı alanına göre $(u,v)$ konumundaki ve $k'$. haritadaki girdisi arasındaki ağırlık.

Aynı $(i,j)$ konumundaki farklı haritaların hücreleri önceki katmanın **tam olarak aynı** piksellerine bağlıdır. Parametre sayısı $(f_h f_w f_{n'}+1)\cdot f_n$; görüntü boyutundan bağımsızdır.
:::

**Ağırlık paylaşımı** modelin parametre sayısını büyük ölçüde azaltır, ama asıl kazanç **genellemedir**: CNN bir deseni bir konumda tanımayı öğrendiğinde onu her konumda tanıyabilir. Sıradan bir DNN bir deseni bir konumda tanımayı öğrendiğinde onu yalnız **o** konumda tanır.

Aşağıdaki deneyde 1-B konvolüsyonda çekirdek boyu, adım, dolgu ve genişlemenin çıktı boyunu ve şeklini nasıl değiştirdiğini gör.

::lab l1

**1×1 konvolüsyon.** $K=1$'de alıcı alan tek bir pikseldir ama tüm kanallara uzanır: her konumda kanallar arası bir doğrusal karışım (piksel başına tam bağlı katman). Kanal sayısını ucuza değiştirmek için kullanılır (Inception, ResNet darboğazı; M4.2, M4.8).

### Bellek ve hesap maliyeti {#maliyet}

CNN'lerin bir sorunu konvolüsyon katmanlarının çok **bellek** istemesidir; özellikle eğitimde, çünkü geri yayılımın ters geçişi ileri geçişte hesaplanan tüm ara değerlere ihtiyaç duyar.

:::örnek Ders slaytındaki hesap
5×5 filtreli, adım 1 ve "same" dolgulu, 200×200 boyutunda 256 öznitelik haritası üreten bir katman; girdi 200×200 RGB.
- **Parametre:** $(5\cdot5\cdot3+1)\cdot256=19\,456$ — tam bağlı bir katmana göre çok az.
- **Hesap:** 256 haritanın her birinde 200×200 nokta, her biri $5\cdot5\cdot3=75$ girdinin ağırlıklı toplamı: $256\cdot200\cdot200\cdot75=768$ milyon kayan nokta çarpması. Tam bağlı bir katman kadar kötü değil ama yine yoğun.
- **Karşılaştırma:** 200×200 nöronlu ilk tam bağlı gizli katman, her nöron $200\cdot200\cdot3$ girdiye bağlı: $200^4\cdot3=4.8$ milyar parametre (ve her biri bir çarpma). Bu yapılandırma için 200×200 genişlik bile yetersizdir — normalde daha geniş olurdu.
- **Bellek:** öznitelik haritaları 32 bitlik sayılarsa çıktı $256\cdot200\cdot200\cdot32\approx328$ milyon bit ≈ 40 MB; 32 örneklik bir batch'te bu katman tek başına ≈ 1.3 GB.
:::

**Çıkarımda** bir katmanın belleği bir sonraki katman hesaplanır hesaplanmaz serbest bırakılabilir; ardışık iki katmanın gerektirdiği kadar bellek yeter. Ama bu fazla iyimserdir (sürekli ayır–bırak kendisi darboğaz olabilir); daha muhafazakâr bir tahmin: tüm ağ parametrelerinin tek kopyası + en büyük katman çıktısı. **Eğitimde** ileri geçişte hesaplanan her şey ters geçiş için saklanmalıdır: gereken bellek (en az) tüm katmanların toplamıdır. Bellek yetmezse: (1) mini-batch'i küçült, (2) adımla boyutu düşür, (3) birkaç katmanı çıkar, (4) 32 yerine 16 bitlik sayılar kullan, (5) CNN'i birden çok cihaza dağıt. (Ayrıca: aktivasyonları saklamak yerine geri geçişte yeniden hesaplamak — gradient checkpointing.)

::lab l3

### im2col: konvolüsyonu matris çarpımına indirgemek {#im2col}

Altı iç içe döngüyle yazılmış bir konvolüsyon doğru ama yavaştır. GPU'lar ve BLAS kütüphaneleri ise büyük matris çarpımlarında son derece verimlidir. **im2col** (image to column) konvolüsyonu tek bir matris çarpımına dönüştürür:

1. Her çıktı konumunun alıcı alanını ($C\times K\times K$ sayı) **bir sütun** olarak düzleştir. $H_{\text{out}}W_{\text{out}}$ konum → $C K^2\times H_{\text{out}}W_{\text{out}}$ boyutlu bir matris $X_{\text{col}}$.
2. Filtreleri satır olarak diz: $W_{\text{row}}$, $F\times CK^2$.
3. $Y=W_{\text{row}}\,X_{\text{col}}+b$, $F\times H_{\text{out}}W_{\text{out}}$; yeniden şekillendirince $F\times H_{\text{out}}\times W_{\text{out}}$.

Bedeli **bellek**: adım 1'de her girdi pikseli yaklaşık $K^2$ farklı pencerede yer alır, dolayısıyla $X_{\text{col}}$ girdiden ≈ $K^2$ kat büyüktür. PyTorch'ta `F.unfold` im2col'u, `F.fold` tersini yapar.

:::türetim Geri geçiş: dW, dX ve col2im
İleri geçiş $Y=W_{\text{row}}X_{\text{col}}$ bir matris çarpımıdır; M0.2'deki kurallarla, $G=\partial L/\partial Y$ için:
$$\frac{\partial L}{\partial W_{\text{row}}}=G\,X_{\text{col}}^\top,\qquad \frac{\partial L}{\partial X_{\text{col}}}=W_{\text{row}}^\top G .$$
$X_{\text{col}}$ girdinin **kopyalarından** oluştuğu için (bir piksel birçok sütunda), girdinin gradyanı o pikselin tüm kopyalarının gradyanlarının **toplamıdır**: $\partial L/\partial X=\mathrm{col2im}\big(W_{\text{row}}^\top G\big)$. col2im, im2col'un **eşleniğidir** (adjoint): her $M$ ve $X$ için $\langle\mathrm{im2col}(X),M\rangle=\langle X,\mathrm{col2im}(M)\rangle$ — kopyalamanın eşleniği toplamadır (M3.3'teki yayın–toplama ikiliğinin aynısı).

**Konvolüsyon diliyle** (1-B, adım 1, dolgusuz korelasyon $y[i]=\sum_u w[u]\,x[i+u]$):
$$\frac{\partial L}{\partial x[j]}=\sum_i\frac{\partial L}{\partial y[i]}\,w[j-i],\qquad \frac{\partial L}{\partial w[u]}=\sum_i\frac{\partial L}{\partial y[i]}\,x[i+u].$$
Girdi gradyanı, $dy$'nin ($K-1$ sıfırla dolgulanmış) **çevrilmiş** çekirdekle korelasyonu, yani "tam" (full) konvolüsyondur; ağırlık gradyanı girdinin $dy$ ile korelasyonudur. Konvolüsyonun geri geçişi de bir konvolüsyondur — transpoze konvolüsyonun (M4.10) kaynağı budur.
:::

Aşağıdaki deneyde küçük bir iki kanallı girdinin im2col matrisini gör. Bir sütuna tıkla: girdideki penceresi ve o sütunun iki filtreyle çarpımı vurgulanır. Bir girdi hücresine tıkla: matristeki tüm kopyaları vurgulanır — col2im'in geri geçişte o hücreye kaç terim topladığı.

::lab l4

### Havuzlama katmanları {#havuzlama}

Havuzlama (pooling) katmanının amacı, hesap yükünü, bellek kullanımını ve parametre sayısını (dolayısıyla aşırı uyum riskini) azaltmak için girdiyi **alt örneklemektir** (küçültmek). Konvolüsyon katmanlarındaki gibi her havuzlama nöronu önceki katmanın sınırlı sayıda nöronuna (küçük dikdörtgen bir alıcı alan) bağlıdır; boyutu, adımı ve dolgu türünü yine tanımlarız. Ama havuzlama nöronunun **ağırlığı yoktur**: girdilerini maksimum ya da ortalama gibi bir toplama fonksiyonuyla birleştirir.

En yaygını **max havuzlamadır**. 2×2 çekirdek, adım 2, dolgu yok: her alıcı alandaki yalnız en büyük girdi bir sonraki katmana geçer, diğerleri atılır (örneğin 1, 5, 3, 2 değerli pencereden yalnız 5). Adım 2 olduğundan çıktının yüksekliği ve genişliği girdinin yarısıdır (dolgu yoksa aşağı yuvarlanır). Havuzlama genelde her girdi kanalında **bağımsız** çalışır; çıktı derinliği girdi derinliğine eşittir.

**Değişmezlik.** Max havuzlama hesap, bellek ve parametreyi azaltmanın yanında küçük kaymalara bir miktar **değişmezlik** kazandırır. A görüntüsünü 2×2 çekirdek ve adım 2 ile havuzla; B, A'nın bir piksel sağa kaymışı, C iki piksel sağa kaymışı olsun. A ve B'nin havuz çıktıları çoğu zaman (parlak pikseller pencerelerinin içinde kaldıkça) **aynıdır**: değişmezlik. C'nin çıktısı farklıdır: bir hücre sağa kaymıştır. Birkaç katmanda bir max havuzlama eklemek daha büyük ölçekte de bir miktar kayma değişmezliği verir; ayrıca biraz döndürme ve hafif ölçek değişmezliği sağlar. Tahminin bu ayrıntılara bağlı olmaması gereken görevlerde (sınıflandırma) yararlıdır.

**Dezavantajları.** (1) Çok yıkıcıdır: 2×2 çekirdek ve adım 2 bile çıktıyı iki yönde yarıya, alanı dörtte bire indirir — girdi değerlerinin %75'i atılır. (2) Bazı uygulamalarda değişmezlik **istenmez**. Anlamsal segmentasyonda girdi bir piksel sağa kayarsa çıktı da bir piksel sağa kaymalıdır: amaç değişmezlik değil **eşdeğişirliktir** (girdideki küçük bir değişim çıktıda karşılık gelen küçük bir değişim yaratmalı).

:::türetim Max havuzlamanın geri geçişi
$y=\max(x_1,\dots,x_n)$ ve maksimum tek bir $x_{k^*}$'da ise $\partial y/\partial x_k=1$ ($k=k^*$), aksi hâlde 0. Gradyan yalnız **kazanan** girdiye gider (ReLU'nun "anahtarı" gibi); ileri geçişte argmax konumları saklanır. Eşitlikte (ör. iki eşit maksimum) türev tanımsızdır; kütüphaneler birini seçer. Ortalama havuzlamada her girdi $1/n$ pay alır.
:::

::lab l5

**Global ortalama havuzlama.** Her haritanın **tüm** konumlarının ortalamasını alır: harita başına tek sayı. Ağın sonunda büyük tam bağlı katmanların yerine kullanılır (GoogLeNet, ResNet; M4.2): parametresizdir ve ağı girdi boyutundan bağımsız kılar. PyTorch'ta `nn.AdaptiveAvgPool2d(1)`.

### Tipik bir CNN ve PyTorch karşılıkları {#mimari}

Tipik CNN mimarileri birkaç konvolüsyon katmanı (her biri genelde bir ReLU ile), sonra bir havuzlama katmanı, sonra birkaç konvolüsyon katmanı (+ReLU), sonra yine havuzlama… diye yığılır. Görüntü ağ boyunca **küçülür** ama konvolüsyon katmanları sayesinde **derinleşir** (daha çok öznitelik haritası). Yığının tepesine birkaç tam bağlı katmandan (+ReLU) oluşan sıradan bir ileri beslemeli ağ eklenir; son katman tahmini verir (ör. sınıf olasılıkları için softmax).

Sık yapılan bir hata **çok büyük çekirdekler** kullanmaktır. 5×5'lik bir katman yerine iki 3×3'lük katman üst üste koy: alıcı alan aynıdır (5×5), parametre ve hesap daha azdır ($2\cdot9C^2=18C^2$ vs $25C^2$), arada bir doğrusalsızlık daha vardır ve genellikle daha iyi çalışır (VGG fikri, M4.2). İstisna ilk konvolüsyon katmanıdır: büyük bir çekirdek (ör. 5×5 ya da 7×7) ve 2 ya da daha büyük adım kullanabilir — girdinin uzamsal boyutunu çok bilgi kaybetmeden düşürür ve girdi genelde yalnız üç kanallı olduğu için pahalı değildir.

| kavram | PyTorch |
|---|---|
| konvolüsyon katmanı | `nn.Conv2d(C, F, kernel_size=K, stride=S, padding=P, dilation=D, padding_mode="zeros")`; ağırlık shape'i `(F, C, K, K)`, bias `(F,)` |
| fonksiyonel biçim | `F.conv2d(x, w, b, stride, padding)`; girdi `(N, C, H, W)` — kanallar önce (NCHW) |
| im2col / col2im | `F.unfold(x, K, stride=S, padding=P)` → `(N, C·K², L)`; `F.fold(cols, (H, W), K, …)` |
| havuzlama | `nn.MaxPool2d(2)` (varsayılan adım = çekirdek), `nn.AvgPool2d(2)`, `nn.AdaptiveAvgPool2d(1)` |
| aynı boyut | `padding="same"` (adım 1) ya da tek K için `padding=K // 2` |

### Özet {#ozet}

- CNN'ler görsel korteksin yerel alıcı alanlarından esinlenir; doğal görüntülerin yerel, konumdan bağımsız ve hiyerarşik yapısını kısmi bağlantı + ağırlık paylaşımıyla kullanır.
- Derin öğrenmedeki "konvolüsyon" çapraz korelasyondur (çekirdek çevrilmez); öğrenilen ağırlıklar için fark önemsiz, elle tasarlanmış filtreler için önemli.
- Çıktı boyu $\lfloor(H+2P-K)/S\rfloor+1$; parametre $(K^2C+1)F$ görüntü boyutundan bağımsız; hesap ve aktivasyon belleği her çıktı konumu için ödenir.
- Konvolüsyon kaymaya eşdeğişirdir (adım 1, kenarlardan uzakta); max havuzlama küçük kaymalara kısmi değişmezlik verir, ama yıkıcıdır ve segmentasyonda istenmez.
- im2col konvolüsyonu $F\times CK^2$ çarpı $CK^2\times H_{\text{out}}W_{\text{out}}$ matris çarpımına indirger (≈ $K^2$ kat bellek); geri geçişte col2im kopyaların gradyanlarını toplar; konvolüsyonun girdi gradyanı çevrilmiş çekirdekle tam konvolüsyondur.
