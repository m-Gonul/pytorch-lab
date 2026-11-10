### İlk gerçek zamanlı yüz dedektörü {#giris}

Paul Viola ve Michael Jones 2001'de (CVPR) yayımladıkları yöntemle yüz tespitini ilk kez gerçek zamanlı yaptılar: makalede 384 × 288 görüntülerde sıradan bir masaüstü işlemcide saniyede yaklaşık 15 kare bildiriliyor. Yöntemin hızı tek bir fikirden değil, birbirini tamamlayan üç fikirden gelir:

1. **Haar benzeri öznitelikler ve integral görüntü:** dikdörtgen toplamlarının farkları, boyutlarından bağımsız sabit sayıda bellek erişimiyle hesaplanır.
2. **AdaBoost:** yüz binlerce olası öznitelikten ayırt edici birkaç yüzünü seçer ve onları ağırlıklı bir oylamayla güçlü bir sınıflandırıcıda birleştirir.
3. **Sınıflandırıcı kaskadı:** pencerelerin büyük çoğunluğu yüz değildir; bunları birkaç öznitelikle erkenden reddeden, art arda dizilmiş aşamalar ortalama maliyeti onlarca kat düşürür.

Tespit bir **kayan pencere** problemidir: 24 × 24'lük bir pencere görüntünün her konumunda ve her ölçeğinde "yüz mü, değil mi?" diye sınıflandırılır. 640 × 480'lik bir karede bu, milyonun üstünde pencere demektir (aşağıda sayıyoruz); her pencereye ayrılabilecek süre mikrosaniyelerle ölçülür. Bu modülün geri kalanı her fikrin bu bütçeye nasıl sığdığını hesapla gösteriyor.

### Haar benzeri öznitelikler {#haar}

Bir Haar benzeri öznitelik, pencere içindeki bitişik, eş boyutlu dikdörtgenlerin piksel toplamları arasındaki farktır. Adını Haar dalgacığından alır: en basit Haar fonksiyonu yarısı $+1$, yarısı $-1$ olan bir basamaktır ve Papageorgiou, Oren ve Poggio (1998) bu fonksiyonların 2-B karşılıklarını nesne tespitinde öznitelik olarak kullanmıştı. Viola–Jones üç aile kullanır:

- **İki dikdörtgen** (kenar öznitelikleri): yan yana ya da üst üste iki dikdörtgenin farkı; bir yönde parlaklık sıçramasını ölçer.
- **Üç dikdörtgen** (çizgi öznitelikleri): ortadaki dikdörtgenden iki dış dikdörtgenin toplamı çıkarılır; ince açık ya da koyu bir bandı ölçer.
- **Dört dikdörtgen:** çapraz çiftlerin farkı (dama tahtası); köşegen yapıları ölçer.

Bu öznitelikler yüzün düzenli karşıtlıklarını yakalar: göz bölgesi altındaki yanaklardan koyudur (dikey iki dikdörtgen), burun köprüsü iki yanındaki gözlerden açıktır (yatay üç dikdörtgen), ağız çevresinden koyu bir yatay banttır. Viola ve Jones'un AdaBoost'unun ilk seçtiği iki öznitelik tam da ilk ikisidir. Bir öznitelik, ders görselindeki gibi bir yüz yamasının üzerine konduğunda tek bir sayı üretir; tek başına zayıf bir ayırıcıdır, ama hesaplaması neredeyse bedavadır.

:::uyarı Kaynak düzeltmesi (slayt 4): dört dikdörtgenli öznitelik "merkez-çevre" değildir
Slayt dört dikdörtgenli özniteliği "center-surround" olarak adlandırıyor. Viola–Jones'un dört dikdörtgenli özniteliği çapraz dikdörtgen çiftlerinin farkıdır (dama tahtası; slayttaki çizim de böyle) ve köşegen yapıya duyarlıdır. Merkez-çevre öznitelikleri (bir dikdörtgenin çevresindeki halkadan farkı) ve 45° döndürülmüş öznitelikler, Lienhart ve Maydt'ın 2002'de genişlettiği kümeye aittir; OpenCV'nin `haarcascade_frontalface_alt` gibi dosyaları bu genişletilmiş kümeyle eğitilmiştir.
:::

**Bir özniteliğin ayırıcılığı.** Bir öznitelik değeri $f(x)$ ile en basit sınıflandırıcı bir **karar kütüğüdür** (decision stump): $h(x)=1$ eğer $p\,f(x)<p\,\theta$, değilse 0; $\theta$ eşik, $p\in\{\pm1\}$ eşitsizliğin yönüdür (polarite). En iyi $(\theta,p)$ çifti, örnekler öznitelik değerine göre bir kez sıralanıp kümülatif toplamlarla taranarak $O(n\log n)$'de bulunur: sıralı listede $k$. örneğin önüne konan bir eşik için hata, eşiğin altındaki negatiflerin ve üstündeki pozitiflerin ağırlıkları toplamıdır (ya da polarite tersse tersi), ve bu iki toplam liste boyunca birer toplamayla güncellenir.

Aşağıdaki deney 24 × 24 yapay yüz yamaları (koyu göz bandı, açık yanaklar, açık burun köprüsü, koyu ağız; değişken konum, kontrast ve aydınlatma) ile yüz olmayan yamalar (dalgalı doku, kenarlar, düz ve gürültülü bölgeler) üretiyor ve seçtiğin tek bir özniteliğin iki sınıftaki dağılımını ve en iyi kütüğün hatasını ölçüyor. Göz bandını yanaklarla karşılaştıran dikey iki dikdörtgenli öznitelik %13.3 hata verirken, aynı bölgede yatay (sol − sağ) öznitelik %37.5'te kalıyor: yüz yatayda neredeyse simetrik olduğu için sol ile sağ arasındaki fark az bilgi taşır. Burun köprüsünü ölçen üç dikdörtgenli öznitelik %9.25'e iniyor.

::lab l1

### Integral görüntü {#integral}

Bir öznitelik doğrudan hesaplanırsa maliyeti kapsadığı piksel sayısıyla büyür: 18 × 8'lik iki dikdörtgenli bir öznitelik 144 piksel okur. **Integral görüntü** (summed-area table; Crow 1984) bunu sabit maliyete indirir:
$$ii(y,x)=\sum_{y'<y}\sum_{x'<x}f(y',x').$$
$ii$, $(y,x)$'in solundaki ve üstündeki bütün piksellerin toplamıdır; boyutu $(H+1)\times(W+1)$'dir, ilk satır ve sütun sıfırdır. Tek geçişte iki toplayıcıyla hesaplanır: satır içi birikimli toplam $s(y,x)=s(y,x-1)+f(y,x)$ ve $ii(y+1,x+1)=ii(y,x+1)+s(y,x)$. Her piksel bir kez okunur.

:::türetim Bir dikdörtgenin toplamı neden dört erişimle bulunur?
Köşeleri $A$ (sol üst), $B$ (sağ üst), $C$ (sol alt), $D$ (sağ alt) olan $R$ dikdörtgenini düşün; $ii(D)$, $D$'nin sol üstündeki bütün bölgenin toplamıdır. Bu bölge dört parçaya ayrılır: $R$, $R$'nin üstündeki şerit $U$, solundaki şerit $L$ ve sol üst köşedeki bölge $Q$. Tanımdan $ii(D)=R+U+L+Q$, $ii(B)=U+Q$, $ii(C)=L+Q$, $ii(A)=Q$. Dolayısıyla
$$ii(D)-ii(B)-ii(C)+ii(A)=R.$$
Bu, içerme–dışarma ilkesidir: $Q$ iki kez çıkarıldığı için bir kez geri eklenir. Sonuç dikdörtgenin boyutuna bağlı değildir. Bitişik dikdörtgenler köşe paylaştığı için iki dikdörtgenli bir öznitelik 6, üç dikdörtgenli 8, dört dikdörtgenli 9 erişimle hesaplanır.
:::

Ders slaytındaki 5 × 5 örnekte vurgulanan 2 × 2 bölgenin toplamı ($3+2+5+4=14$) integral görüntüden $46-22-20+10=14$ olarak okunur.

<figure class="fig wide"><svg viewBox="0 0 640 340" role="img" aria-label="Beş Haar öznitelik türü, her birinin 24 × 24 penceredeki sayısı ve integral görüntüde dikdörtgen toplamının dört erişimle hesaplanması"><rect x="20.0" y="20.0" width="30.0" height="60.0" class="soft-a" stroke="currentColor" stroke-width="1"/><text x="35.0" y="55.0" text-anchor="middle" class="sm" style="">+</text><rect x="50.0" y="20.0" width="30.0" height="60.0" class="ca-f" stroke="currentColor" stroke-width="1"/><text x="65.0" y="55.0" text-anchor="middle" class="sm" style="fill:#fff">−</text><text x="50" y="98" text-anchor="middle" class="sm">2 yatay (kenar)</text><text x="50" y="114" text-anchor="middle" class="sm">43 200 öznitelik</text><text x="50" y="130" text-anchor="middle" class="sm">6 erişim</text><rect x="144.0" y="20.0" width="60.0" height="30.0" class="soft-a" stroke="currentColor" stroke-width="1"/><text x="174.0" y="40.0" text-anchor="middle" class="sm" style="">+</text><rect x="144.0" y="50.0" width="60.0" height="30.0" class="ca-f" stroke="currentColor" stroke-width="1"/><text x="174.0" y="70.0" text-anchor="middle" class="sm" style="fill:#fff">−</text><text x="174" y="98" text-anchor="middle" class="sm">2 dikey (kenar)</text><text x="174" y="114" text-anchor="middle" class="sm">43 200 öznitelik</text><text x="174" y="130" text-anchor="middle" class="sm">6 erişim</text><rect x="268.0" y="20.0" width="20.0" height="60.0" class="ca-f" stroke="currentColor" stroke-width="1"/><text x="278.0" y="55.0" text-anchor="middle" class="sm" style="fill:#fff">−</text><rect x="288.0" y="20.0" width="20.0" height="60.0" class="soft-a" stroke="currentColor" stroke-width="1"/><text x="298.0" y="55.0" text-anchor="middle" class="sm" style="">+</text><rect x="308.0" y="20.0" width="20.0" height="60.0" class="ca-f" stroke="currentColor" stroke-width="1"/><text x="318.0" y="55.0" text-anchor="middle" class="sm" style="fill:#fff">−</text><text x="298" y="98" text-anchor="middle" class="sm">3 yatay (çizgi)</text><text x="298" y="114" text-anchor="middle" class="sm">27 600 öznitelik</text><text x="298" y="130" text-anchor="middle" class="sm">8 erişim</text><rect x="392.0" y="20.0" width="60.0" height="20.0" class="ca-f" stroke="currentColor" stroke-width="1"/><text x="422.0" y="35.0" text-anchor="middle" class="sm" style="fill:#fff">−</text><rect x="392.0" y="40.0" width="60.0" height="20.0" class="soft-a" stroke="currentColor" stroke-width="1"/><text x="422.0" y="55.0" text-anchor="middle" class="sm" style="">+</text><rect x="392.0" y="60.0" width="60.0" height="20.0" class="ca-f" stroke="currentColor" stroke-width="1"/><text x="422.0" y="75.0" text-anchor="middle" class="sm" style="fill:#fff">−</text><text x="422" y="98" text-anchor="middle" class="sm">3 dikey (çizgi)</text><text x="422" y="114" text-anchor="middle" class="sm">27 600 öznitelik</text><text x="422" y="130" text-anchor="middle" class="sm">8 erişim</text><rect x="516.0" y="20.0" width="30.0" height="30.0" class="soft-a" stroke="currentColor" stroke-width="1"/><text x="531.0" y="40.0" text-anchor="middle" class="sm" style="">+</text><rect x="546.0" y="20.0" width="30.0" height="30.0" class="ca-f" stroke="currentColor" stroke-width="1"/><text x="561.0" y="40.0" text-anchor="middle" class="sm" style="fill:#fff">−</text><rect x="516.0" y="50.0" width="30.0" height="30.0" class="ca-f" stroke="currentColor" stroke-width="1"/><text x="531.0" y="70.0" text-anchor="middle" class="sm" style="fill:#fff">−</text><rect x="546.0" y="50.0" width="30.0" height="30.0" class="soft-a" stroke="currentColor" stroke-width="1"/><text x="561.0" y="70.0" text-anchor="middle" class="sm" style="">+</text><text x="546" y="98" text-anchor="middle" class="sm">4 (çapraz)</text><text x="546" y="114" text-anchor="middle" class="sm">20 736 öznitelik</text><text x="546" y="130" text-anchor="middle" class="sm">9 erişim</text><text x="20" y="150" class="sm">24 × 24 pencerede her tür için bütün konum ve boyutlar: toplam 162 336 öznitelik (hesaplandı).</text><path d="M40,175 L40,329" class="mute"/><path d="M62,175 L62,329" class="mute"/><path d="M84,175 L84,329" class="mute"/><path d="M106,175 L106,329" class="mute"/><path d="M128,175 L128,329" class="mute"/><path d="M150,175 L150,329" class="mute"/><path d="M172,175 L172,329" class="mute"/><path d="M194,175 L194,329" class="mute"/><path d="M216,175 L216,329" class="mute"/><path d="M40,175 L216,175" class="mute"/><path d="M40,197 L216,197" class="mute"/><path d="M40,219 L216,219" class="mute"/><path d="M40,241 L216,241" class="mute"/><path d="M40,263 L216,263" class="mute"/><path d="M40,285 L216,285" class="mute"/><path d="M40,307 L216,307" class="mute"/><path d="M40,329 L216,329" class="mute"/><rect x="106" y="219" width="88" height="66" class="soft-b" stroke="currentColor" stroke-width="1.5"/><circle cx="106" cy="219" r="4.5" class="cb-f"/><text x="94" y="213" class="sm">A</text><circle cx="194" cy="219" r="4.5" class="cb-f"/><text x="201" y="213" class="sm">B</text><circle cx="106" cy="285" r="4.5" class="cb-f"/><text x="94" y="279" class="sm">C</text><circle cx="194" cy="285" r="4.5" class="cb-f"/><text x="201" y="279" class="sm">D</text><text x="150" y="256.4" text-anchor="middle" class="sm">R</text><text x="250" y="187" class="sm">ii(y, x): solundaki ve üstündeki piksellerin toplamı,</text><text x="250" y="206" class="sm">tek geçişte satır toplamıyla birlikte hesaplanır.</text><text x="250" y="225" class="sm">R'nin toplamı = ii(D) − ii(B) − ii(C) + ii(A):</text><text x="250" y="244" class="sm">boyuttan bağımsız 4 erişim.</text><text x="250" y="263" class="sm">Köşe paylaşımı: 2, 3, 4 dikdörtgen → 6, 8, 9 erişim.</text><text x="250" y="282" class="sm">f²'nin integral görüntüsüyle σ² = E[f²] − μ²</text><text x="250" y="301" class="sm">(pencere başına 8 erişim daha).</text><text x="250" y="320" class="sm">Ölçek: öznitelik büyütülür, piramit gerekmez.</text></svg><figcaption>Şekil 1 — Haar öznitelikleri açık (+) ve koyu (−) dikdörtgen toplamlarının farkıdır; integral görüntü her dikdörtgeni boyutundan bağımsız 4 erişimle verir (sayılar hesaplandı).</figcaption></figure>

**Ölçek.** Dikdörtgen toplamının maliyeti boyuttan bağımsız olduğu için Viola–Jones, görüntü piramidi kurmak yerine **dedektörü** büyütür: ölçek $s$'de bütün dikdörtgen koordinatları $s$ ile çarpılıp yuvarlanır, öznitelik yine 6–9 erişimle hesaplanır. Görüntüyü her ölçekte yeniden örneklemek ve integral görüntüsünü yeniden hesaplamak gerekmez.

**Pencere varyans normalizasyonu.** Eğitim örnekleri ortalaması 0, varyansı 1 olacak biçimde normalize edilir; tespitte de her pencerenin aynı biçimde normalize edilmesi gerekir. Pikselleri tek tek normalize etmek pahalı olurdu; bunun yerine iki integral görüntü tutulur: $f$ ve $f^2$. Bir pencerenin ortalaması $\mu=\Sigma f/N$, varyansı $\sigma^2=\Sigma f^2/N-\mu^2$; ikisi de 4'er erişimle gelir. Normalizasyon özniteliğe uygulanır, piksellere değil.

:::türetim Normalizasyon bir özniteliği nasıl değiştirir?
Normalize pencere $g=(f-\mu)/\sigma$ olsun. Bir öznitelik, ağırlıkları $c_k$ olan dikdörtgen toplamlarının doğrusal birleşimidir: $F(f)=\sum_kc_k\,\Sigma_{R_k}f$. Doğrusallıktan
$$F(g)=\frac{F(f)-\mu\sum_kc_k|R_k|}{\sigma}.$$
İki ve dört dikdörtgenli özniteliklerde ağırlıkların alanlarla toplamı sıfırdır ($wh-wh=0$): ortalama terimi düşer, $F(g)=F(f)/\sigma$. Bu özniteliklerde normalizasyon yalnız bir bölme demektir ve **ortalama** (ofset) zaten özniteliği etkilemez; normalizasyonun kazandırdığı, **kontrast** (kazanç) değişimine değişmezliktir. Üç dikdörtgenli öznitelikte (orta − iki dış) ise $\sum c_k|R_k|=wh-2wh=-wh$: $F(g)=(F(f)+\mu\,wh)/\sigma$. Normalizasyonsuz ham değer pencerenin parlaklığını da taşır. Lienhart ve Maydt bu yüzden ağırlıkları alanları dengeleyecek biçimde seçer (bütün dikdörtgen $-1$, orta $+3$): öznitelik sıfır toplamlı olur.
:::

Lab l1 bunu ölçüyor. Dikey iki dikdörtgenli öznitelikte normalizasyon hatayı %16.3'ten %13.3'e indiriyor. Aydınlatma değişkenliği sıfırken bile fark sürüyor (%16'ya %13): yamaların kontrastı da değişkendir ve normalizasyon onu da siliyor. Burun köprüsünü ölçen üç dikdörtgenli öznitelikte normalizasyonsuz hata %29, normalize %9.25: türetimdeki ortalama terimi özniteliğe parlaklığı karıştırıyor.

:::uyarı Integral görüntünün veri tipi
640 × 480'lik 8 bitlik bir görüntüde $ii$'nin en büyük değeri $640\cdot480\cdot255=78\,336\,000$'dir ve 32 bitlik tamsayıya sığar. Karelerin integral görüntüsü ise $640\cdot480\cdot255^2\approx2.0\times10^{10}$'a ulaşır ve $2^{31}\approx2.1\times10^9$'u aşar. OpenCV `cv2.integral2` bu yüzden kare toplamını `float64` ile döndürür. NumPy'da `np.cumsum` uint8 girdiyi `uint64`'e yükseltir, ama `astype(np.int32)` ile sabitlenmiş bir dizide taşma sessizce olur.
:::

### Öznitelik seçimi problemi {#secim}

24 × 24'lük bir pencerede her türün olası bütün konum ve boyutları sayılırsa öznitelik kümesi çok büyüktür. Temel boyutu $s_x\times s_y$ olan bir tür için (iki yatay dikdörtgen: $2\times1$), genişlik $s_x$'in, yükseklik $s_y$'nin katlarıdır ve her boyut için $(25-w)(25-h)$ konum vardır:
$$N(s_x,s_y)=\sum_{a=1}^{\lfloor24/s_x\rfloor}(25-a\,s_x)\sum_{b=1}^{\lfloor24/s_y\rfloor}(25-b\,s_y).$$
İki dikdörtgenli yatay tür için iç toplamlar $\sum_{a=1}^{12}(25-2a)=144$ ve $\sum_{b=1}^{24}(25-b)=300$, yani 43 200; üç dikdörtgenli türlerde $92\cdot300=27\,600$, dört dikdörtgenlide $144^2=20\,736$. Toplam: $2\cdot43\,200+2\cdot27\,600+20\,736=$ **162 336**.

:::sezgi 160 000 mi, 180 000 mi?
Slayttaki "160 000'den fazla" sayısı, yöntemin 2004 tarihli dergi sürümüyle (IJCV) uyumludur ve yukarıdaki 162 336 ile tutar. 2001 konferans makalesi ise "180 000'den fazla" der. Fark sayım kurallarından gelir (hangi türlerin ve boyutların dahil edildiği); scikit-image `haar_like_feature_coord(24, 24)` aynı beş türle 162 336 döndürür.
:::

Bu kadar özniteliği her pencerede hesaplamak, her biri 6–9 erişim olsa bile, olanaksızdır: Şekil 1'deki sayıyı 640 × 480'de yaklaşık 1.3 milyon pencereyle çarpınca kare başına $2\times10^{11}$ öznitelik eder. Soru şudur: yüzleri en iyi ayıran birkaç yüz öznitelik hangileridir? Bu bir **öznitelik seçimi** problemidir ve Viola–Jones onu sınıflandırıcıyı eğitirken çözer.

### AdaBoost: seçim ve birleştirme {#adaboost}

**Boosting**, her biri rastgele tahminden biraz iyi olan **zayıf sınıflandırıcıları** ağırlıklı bir oylamayla güçlü bir sınıflandırıcıda birleştirir (Freund ve Schapire 1997). Viola–Jones'taki kurnazlık, zayıf sınıflandırıcıyı **tek bir özniteliğe** bağlamaktır: her tur bir karar kütüğü seçer, bu da bir öznitelik seçmek demektir. $T$ tur sonunda güçlü sınıflandırıcı $T$ öznitelik kullanır. Bu bir makine öğrenmesi görevidir ve görüntü işleme burada yalnız öznitelikleri sağlar.

Eğitim örnekleri $(x_i,y_i)$, $y_i\in\{0,1\}$; $m$ negatif, $l$ pozitif:

1. Başlangıç ağırlıkları: negatiflere $w_{1,i}=\frac1{2m}$, pozitiflere $\frac1{2l}$. İki sınıfın toplam ağırlığı eşittir.
2. Her $t=1,\dots,T$ turunda:
    - Ağırlıkları normalize et: $w_{t,i}\leftarrow w_{t,i}/\sum_jw_{t,j}$.
    - Her öznitelik için ağırlıklı hatası en küçük kütüğü bul: $\epsilon_j=\sum_iw_{t,i}\,|h_j(x_i)-y_i|$.
    - En küçük $\epsilon_t$'yi veren kütüğü $h_t$ olarak seç.
    - Ağırlıkları güncelle: $w_{t+1,i}=w_{t,i}\,\beta_t^{\,1-e_i}$, burada $e_i=0$ doğru, $e_i=1$ yanlış sınıflanan örnek için ve $\beta_t=\epsilon_t/(1-\epsilon_t)$.
3. Güçlü sınıflandırıcı:
$$C(x)=\begin{cases}1,&\sum_{t=1}^T\alpha_th_t(x)\ge\frac12\sum_{t=1}^T\alpha_t\\0,&\text{değilse}\end{cases},\qquad \alpha_t=\log\frac1{\beta_t}.$$

$\epsilon_t<\frac12$ olduğu sürece $\beta_t<1$: doğru sınıflananların ağırlığı küçülür, normalizasyondan sonra yanlış sınıflananların payı **artar**. Sonraki tur bu zor örneklere odaklanan bir öznitelik seçer. Ağırlık $\alpha_t$ küçük hatalı kütüklere büyük oy verir: $\epsilon=0.2$ için $\beta=0.25$, $\alpha=\ln4=1.386$.

:::uyarı Kaynak düzeltmesi (slayt 9): başlangıç ağırlıkları her zaman eşit değildir
Slayt "bütün örneklere eşit ağırlık" ile başlıyor. Bu, özgün AdaBoost'un (Freund–Schapire) başlangıcıdır. Viola–Jones sınıfları dengeler: pozitiflere $1/2l$, negatiflere $1/2m$. İkisi yalnız $l=m$ iken aynıdır. Yüz tespitinde negatifler pozitiflerden çok daha fazladır (makalede yaklaşık 5 000 yüz, aynalanmışlarıyla iki katı, ve her aşama için yüz içermeyen görüntülerden yeniden toplanan 10 000 negatif). Eşit örnek ağırlığıyla ilk kütükler negatifleri doğru sınıflamaya yönelir. Slayttaki "yanlış sınıflananların ağırlığını artır" ifadesi de sonuç olarak doğrudur. Ancak Viola–Jones bunu doğruların ağırlığını $\beta$ ile **azaltıp** normalize ederek yapar; iki işlem normalizasyondan sonra aynı dağılımı verir.
:::

:::türetim Viola–Jones biçimi özgün AdaBoost ile neden aynıdır?
Özgün ayrık AdaBoost etiketleri ve çıktıları $\pm1$ alır: $H(x)=\mathrm{sign}\sum_t a_t\,h'_t(x)$, $a_t=\frac12\ln\frac{1-\epsilon_t}{\epsilon_t}$; ağırlıklar doğrularda $e^{-a_t}$, yanlışlarda $e^{a_t}$ ile çarpılır.

*Karar aynı:* $h'=2h-1$ koyalım. $\sum_t\alpha_th_t\ge\frac12\sum_t\alpha_t$ eşitsizliği $\sum_t\alpha_t(2h_t-1)\ge0$, yani $\sum_t\alpha_th'_t\ge0$ ile denktir. $\alpha_t=\ln\frac{1-\epsilon_t}{\epsilon_t}=2a_t$ olduğu için işaret aynıdır.

*Güncelleme aynı:* özgün kuralda yanlış/doğru ağırlık oranı $e^{a_t}/e^{-a_t}=e^{2a_t}=\frac{1-\epsilon_t}{\epsilon_t}$. Viola–Jones'ta oran $1/\beta_t$, yani aynısı. Normalizasyon ortak çarpanı siler.

*Bir sonuç:* güncellemeden sonra doğruların toplam ağırlığı $(1-\epsilon_t)\beta_t=\epsilon_t$, yanlışlarınki $\epsilon_t$'dir. Seçilen kütük yeni dağılımda tam **%50 hatalıdır**, bu yüzden bir sonraki tur onu (aynı eşik ve polariteyle) yeniden seçmez.

*Eğitim hatası sınırı:* Freund ve Schapire, güçlü sınıflandırıcının eğitim hatasının $\prod_t2\sqrt{\epsilon_t(1-\epsilon_t)}$ ile sınırlı olduğunu gösterir. $\epsilon_t=\frac12-\gamma_t$ yazılırsa bu $\prod_t\sqrt{1-4\gamma_t^2}\le\exp(-2\sum_t\gamma_t^2)$ olur. Her kütük rastgeleden biraz iyiyse eğitim hatası turlarla **üstel** düşer. $\epsilon=0.2$ için tur başına çarpan 0.8'dir. Sınır testi değil eğitimi bağlar; lab l2'de eğitim hatası sıfıra inerken test hatası %4–5'te kalıyor.
:::

**Hesap maliyeti.** Her turda bütün öznitelikler için en iyi kütük aranır. Öznitelik değerleri bütün örnekler için bir kez hesaplanıp sıralanırsa (sıralama ağırlığa bağlı değildir), her tur her öznitelik için yalnız sıralı listede bir tarama gerektirir: $O(Kn)$, $K$ öznitelik ve $n$ örnek için. 162 336 öznitelik ve 15 000 örnekte bu tur başına yaklaşık $2.4\times10^9$ adımdır ve 200 tur saatler sürer. Makalenin eğitimi haftalar sürmüştür; sonraki çalışmalar öznitelik alt örneklemesi ve önbellekli sıralamayla hızlandırır.

Aşağıdaki deney rastgele bir Haar öznitelik havuzu (normalize pencerelerde) üzerinde AdaBoost çalıştırıyor. Yüz olmayanların bir kısmı **zor negatiflerdir**: yüzün parçaları (göz bandı, yanak bandı, burun köprüsü, ağız) rastgele alt kümeler ve kaymalarla. Tek bir öznitelik bunları yüzlerden ayıramaz. Tek AdaBoost'ta ilk kütük %10.3 hata veriyor; 16 turda eğitim hatası %0.83'e, test hatası %16.6'dan %4.74'e iniyor. Aynı deney, aşağıdaki kaskad bölümünde anlatılan biçimde bir kaskad da eğitiyor.

::lab l2

### Sınıflandırıcı kaskadı {#kaskad}

Makalede 200 öznitelikli tek parça bir AdaBoost sınıflandırıcısı %95 tespit oranında 14 084 pencerede 1 yanlış pozitif veriyor: doğruluk iyi, ama her pencerede 200 öznitelik hesaplamak gerçek zaman için çok yavaş. Gözlem şudur: bir görüntüdeki pencerelerin neredeyse hepsi yüz değildir ve çoğu, yüze hiç benzemediği için birkaç öznitelikle reddedilebilir. **Kaskad**, sınıflandırıcıları artan karmaşıklıkta art arda dizer. Bir aşama "hayır" derse pencere hemen reddedilir ve sonraki aşamalara hiç gitmez. Yalnız bütün aşamalardan geçen pencere yüz sayılır. Bu bir **yozlaşmış karar ağacıdır**: her düğümde tek bir "evet" dalı ilerler.

- **İlk aşamalar** az öznitelikli ve basittir: çok yüksek tespit oranıyla (hemen hemen bütün yüzleri geçirir) pencerelerin büyük kısmını eler. Viola–Jones'un ilk aşaması 2 öznitelikle yüzlerin neredeyse hepsini geçirip negatiflerin yaklaşık %50'sini reddeder; ikincisi 10 öznitelikle %80'ini.
- **Sonraki aşamalar** daha çok öznitelik kullanır ve ilk aşamaları geçen, yüze daha çok benzeyen zor pencereleri ayıklar.

**Aşama eşiği.** Her aşama bir AdaBoost sınıflandırıcısıdır, ama eşiği $\frac12\sum\alpha_t$ değildir. Eşik, doğrulama yüzlerinin istenen oranını (ör. %99.5) geçirecek kadar **düşürülür**: bir aşamanın amacı hata en küçüklemek değil, neredeyse hiç yüz kaçırmadan olabildiğince çok negatif elemektir. Yanlış pozitifleri sonraki aşamalar temizler.

**Kaskad aritmetiği.** Aşama $i$'nin tespit oranı $d_i$, yanlış pozitif oranı $f_i$ ise (koşullu oranlar: önceki aşamaları geçenler arasında) $K$ aşamalı kaskad için
$$D=\prod_{i=1}^Kd_i,\qquad F=\prod_{i=1}^Kf_i.$$
10 aşama, $d_i=0.99$, $f_i=0.3$: $D=0.99^{10}=0.904$, $F=0.3^{10}=5.9\times10^{-6}$. Tespit oranı yavaşça, yanlış pozitif oranı üstel düşer (Şekil 2). Bir negatif pencerenin beklenen maliyeti, aşama $i$'nin öznitelik sayısı $n_i$ ve o aşamaya ulaşma olasılığıyla:
$$E[\text{öznitelik}]=n_1+\sum_{i=2}^Kn_i\prod_{j<i}f_j.$$

<figure class="fig wide"><svg viewBox="0 0 640 262" role="img" aria-label="Kaskadın toplam tespit ve yanlış pozitif oranının aşama sayısıyla değişimi (logaritmik eksen) ve örnek bir kaskadın pencere başına beklenen öznitelik sayısı"><path d="M60,24.0 L360,24.0" class="mute"/><text x="54" y="28.0" text-anchor="end" class="sm">1</text><path d="M60,57.3 L360,57.3" class="mute"/><text x="54" y="61.3" text-anchor="end" class="sm">10⁻¹</text><path d="M60,90.7 L360,90.7" class="mute"/><text x="54" y="94.7" text-anchor="end" class="sm">10⁻²</text><path d="M60,124.0 L360,124.0" class="mute"/><text x="54" y="128.0" text-anchor="end" class="sm">10⁻³</text><path d="M60,157.3 L360,157.3" class="mute"/><text x="54" y="161.3" text-anchor="end" class="sm">10⁻⁴</text><path d="M60,190.7 L360,190.7" class="mute"/><text x="54" y="194.7" text-anchor="end" class="sm">10⁻⁵</text><path d="M60,224.0 L360,224.0" class="mute"/><text x="54" y="228.0" text-anchor="end" class="sm">10⁻⁶</text><text x="60.0" y="240" text-anchor="middle" class="sm">0</text><text x="120.0" y="240" text-anchor="middle" class="sm">2</text><text x="180.0" y="240" text-anchor="middle" class="sm">4</text><text x="240.0" y="240" text-anchor="middle" class="sm">6</text><text x="300.0" y="240" text-anchor="middle" class="sm">8</text><text x="360.0" y="240" text-anchor="middle" class="sm">10</text><text x="210.0" y="256" text-anchor="middle" class="sm">aşama sayısı</text><path d="M60.0,24.0 L90.0,24.1 L120.0,24.3 L150.0,24.4 L180.0,24.6 L210.0,24.7 L240.0,24.9 L270.0,25.0 L300.0,25.2 L330.0,25.3 L360.0,25.5" class="ca w2" fill="none"/><path d="M60.0,24.0 L90.0,41.4 L120.0,58.9 L150.0,76.3 L180.0,93.7 L210.0,111.1 L240.0,128.6 L270.0,146.0 L300.0,163.4 L330.0,180.9 L360.0,198.3" class="cb w2" fill="none"/><circle cx="360.0" cy="25.5" r="4" class="ca-f"/><circle cx="360.0" cy="198.3" r="4" class="cb-f"/><text x="385" y="34" class="ta">D = 0.99<tspan baseline-shift="4" font-size="75%">10</tspan> = 0.9044 (tespit)</text><text x="385" y="54" class="tb">F = 0.3<tspan baseline-shift="4" font-size="75%">10</tspan> = 5.90 × 10⁻⁶ (yanlış pozitif)</text><text x="385" y="88" class="sm">Örnek kaskad (aşama boyları):</text><text x="385" y="106" class="sm">2, 10, 25, 25, 50, 50, 50, 75, 100, 200</text><text x="385" y="124" class="sm">toplam 587 öznitelik; negatifin bir aşamayı</text><text x="385" y="142" class="sm">geçme oranı 0.5, 0.2, sonra 0.3.</text><text x="385" y="166" class="sm">Bir negatif pencere için beklenen maliyet</text><text x="385" y="184" class="sm">Σ n<tspan baseline-shift="-3" font-size="75%">i</tspan> × (önceki geçme oranları) = 2 + 5 + 2.5 + …</text><text x="385" y="202" class="sm">= 10.91 öznitelik (tek parça 587 yerine).</text></svg><figcaption>Şekil 2 — Aşama oranları çarpılır: tespit yavaş, yanlış pozitif üstel düşer (hesaplandı). Negatif pencerelerin çoğu ilk iki aşamada elendiği için ortalama maliyet 10.9 özniteliktir; aşama boyları ve geçme oranları örnek amaçlıdır (ilk ikisi Viola–Jones 2004'teki iki aşamalı başlangıca benzer).</figcaption></figure>

**Kaskadın eğitimi.** Kullanıcı aşama başına en büyük $f$ ve en küçük $d$ ile hedef toplam $F_\text{hedef}$'i seçer. Her aşamada AdaBoost'a öznitelik eklenir, her eklemede eşik doğrulama yüzlerinde $d$'yi sağlayacak kadar düşürülür ve yanlış pozitif oranı $f$'nin altına inince aşama biter. Sonraki aşamanın negatifleri, şimdiye kadarki kaskadın **yanlış pozitif verdiği** pencerelerden (yüz içermeyen görüntüler taranarak) toplanır. Bu **zor negatif madenciliğidir** (bootstrapping): her aşama, öncekilerin kandığı örneklerle eğitilir. Dergi sürümündeki son dedektör 38 aşama ve toplam 6 060 öznitelikten oluşur (konferans sürümünde 32 aşama, 4 297 öznitelik), ama bir pencere için ortalama yaklaşık 10 öznitelik hesaplanır.

:::sezgi Lab l2'deki kaskad
Lab l2'de her aşama ayrı bir AdaBoost'tur. Pozitifleri önceki aşamaları geçen eğitim yüzleri, negatifleri 3 000 pencerelik bir havuzdan şimdiye kadarki kaskadı geçenlerdir (zor negatif madenciliği). Aşamaya, eşik yüzlerin en az $d$ oranını geçirecek biçimde düşürüldüğünde negatiflerin en çok $f$'si geçene kadar öznitelik eklenir. Varsayılan ayarda ($d=0.99$, $f=0.4$) 5 aşama sırasıyla 6, 5, 9, 9, 6 öznitelik kullanıyor. Bir negatif pencere ortalama 8.9 öznitelik hesaplatıyor, bir yüz 35'in neredeyse hepsini (33.7). İlk aşamanın tek öznitelikle yetmemesi öğreticidir: kütük çıktısı 0 ya da 1'dir. Yüzlerin %99'unu geçirmek için tek kütüklü aşamanın eşiği sıfıra iner ve hiçbir şey reddedilmez; birkaç kütüğün ağırlıklı toplamı gerekir. Eğitimde $D\ge d^5$ garanti edilir (0.967); testte 0.92'ye düşüyor. 300 yüzle seçilen eşikler alt kuyruğu iyi kestiremez ve kayıplar aşamalarla çarpılır; Viola–Jones eşikleri ayrı doğrulama yüzlerinde ayarlar.
:::

### Algoritma ve çok ölçekli tarama {#tarama}

Tespit adımları:

1. Görüntüyü gri seviyeye çevir.
2. Integral görüntüyü ve kare integral görüntüyü hesapla.
3. Pencereyi her konumda ve her ölçekte kaydır.
4. Her pencerede varyansı bul ve Haar özniteliklerini integral görüntüden hesapla.
5. Pencereyi kaskaddan geçir.
6. Bütün aşamaları geçen pencereleri işaretle ve örtüşenleri birleştir.

Viola–Jones birleştirmede örtüşen tespitleri ayrık kümelere ayırıp her kümenin ortalama kutusunu alır; bugün yaygın olan maksimum olmayanı bastırmadır (NMS, M4.9).

**Kaç pencere?** 640 × 480'de 24 × 24'lük pencereyi bir piksel adımla, ölçeği 1.1 kat artırarak 24'ten 60 piksele kadar taramak 10 ölçek ve toplam yaklaşık **1.33 milyon** pencere eder (ölçek 1'de 281 969, 2.36'da 44 640). Kaskad bir pencerede ortalama 10 öznitelik hesaplarsa, öznitelik başına 6–9 erişimle kare başına yaklaşık $10^8$ bellek erişimi eder. Bu gerçek zamana sığar; tek parça 200 öznitelikli sınıflandırıcıyla sayı 20 kat büyürdü. Uygulamalar büyük ölçeklerde adımı da ölçekle büyütür.

**OpenCV.** Ders örneği OpenCV'nin önceden eğitilmiş `haarcascade_frontalface_default.xml` kaskadını bir videonun her karesinde çalıştırıyor:

- `detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20), maxSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)`.
- `scaleFactor`, ardışık ölçekler arasındaki oran: küçük değer daha çok ölçek, daha iyi kapsama, daha yavaş tarama demektir.
- `minNeighbors`, bir tespitin kabul edilmesi için kaç örtüşen ham tespitin kümelenmesi gerektiği: büyük değer yanlış pozitifleri azaltır ama küçük ya da kısmi yüzleri kaçırabilir.
- `minSize`/`maxSize`, aranacak yüz boyutlarını sınırlar. Bu kaskad 24 × 24'te eğitildiği için 24'ün altındaki `minSize` etkisizdir, `maxSize=(60, 60)` ise yakındaki büyük yüzleri hiç aramaz.
- `CASCADE_SCALE_IMAGE` bayrağı, OpenCV'nin dedektörü büyütmek yerine **görüntüyü küçülttüğü** yolu seçer; makalenin öznitelik ölçekleme fikrinden farklı ama önbellek dostu bir uygulamadır.

Ders görselindeki sokak sahnesi kaskadın tipik davranışını gösterir: kameraya dönük yüzler bulunur, profilden ya da eğik görünenler kaçırılır.

### Artılar ve eksiler {#arti-eksi}

**Artılar.** Gerçek zamanlıdır: 2001 donanımında kare başına onlarca milisaniye, bugün gömülü işlemcilerde bile hızlıdır. Uygulaması basittir: tespit yalnız toplama, çıkarma ve karşılaştırmadır, GPU gerekmez. Kaskad dosyası küçüktür.

**Eksiler.** Yalnız dik ve önden yüzler için eğitilir: yüz düzlem içinde ±15° kadar dönünce ya da profile dönünce performans hızla düşer, profil için ayrı kaskad gerekir. Dikdörtgen öznitelikler eksenlere bağlıdır ve dönmeye değişmez değildir (Lienhart'ın 45° öznitelikleri bunu kısmen hafifletir). Kısmi örtme, aşırı aydınlatma ve küçük yüzlerde zayıftır. Eğitimi uzun sürer. Bugün CNN tabanlı dedektörler (MTCNN 2016, RetinaFace 2020 gibi) doğrulukta açık ara öndedir; Viola–Jones, hızın her şeyden önemli olduğu kısıtlı donanımda ve tarihsel bir başvuru noktası olarak yaşıyor.

:::uyarı Kaynak düzeltmesi (slayt 15): aydınlatma dayanıklılığı nereden geliyor?
Slayt aydınlatma dayanıklılığını "Haar karşıtlık özniteliklerine" bağlıyor. Fark öznitelikleri (iki ve dört dikdörtgenliler) yalnız sabit bir **ofseti** (bütün pencereye eklenen parlaklık) siler. Kontrast değişimi (kazanç) öznitelik değerini doğrudan ölçekler, üç dikdörtgenli öznitelikler ise ofseti de taşır (yukarıdaki türetim). Dayanıklılığın asıl kaynağı **pencere varyans normalizasyonudur**. Lab l1'de normalizasyonu kapatınca dikey iki dikdörtgenli özniteliğin hatası %13.3'ten %16.3'e, üç dikdörtgenlinin %9.25'ten %29'a çıkıyor. Varyans normalizasyonu da yalnız pencere çapında doğrusal bir değişimi düzeltir; yüzün yarısına düşen gölge gibi yerel aydınlatma farklarını gidermez.
:::

### OpenCV ve PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| integral görüntü | `ii, sq = cv2.integral2(gray)` (sq: `float64`); NumPy: `np.pad(f.cumsum(0).cumsum(1), ((1, 0), (1, 0)))` |
| Haar öznitelikleri | `skimage.feature.haar_like_feature(ii, r, c, h, w, feature_type=..., feature_coord=...)` |
| hazır kaskad | `cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")` |
| tespit | `faces = cc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)` |
| kaskad eğitimi | `opencv_traincascade -featureType HAAR -numStages 20 -minHitRate 0.995 -maxFalseAlarmRate 0.5` (OpenCV 3.4 aracı; 4.x dağıtımlarında yok) |
| PyTorch'ta toplu öznitelik | `ii = F.pad(x.cumsum(-1).cumsum(-2), (1, 0, 1, 0))`; köşe indeksleriyle `ii[..., y2, x2] - ii[..., y1, x2] - ...` bütün pencerelerde bir seferde |

### Özet {#ozet}

- Viola–Jones (2001): ilk gerçek zamanlı yüz dedektörü. Üç fikir: Haar öznitelikleri + integral görüntü, AdaBoost ile seçim, kaskad.
- Haar öznitelikleri dikdörtgen toplamlarının farkıdır. 24 × 24'te 5 türle 162 336 öznitelik vardır (43 200 + 43 200 + 27 600 + 27 600 + 20 736).
- Integral görüntü dikdörtgen toplamını 4 erişimle verir (2, 3 ve 4 dikdörtgenli öznitelik için 6, 8 ve 9). Kare integral görüntü varyans normalizasyonunu sağlar; normalizasyon kazanca, sıfır toplamlı olmayan özniteliklerde ofsete de değişmezlik getirir.
- AdaBoost ağırlıkları $1/2l$, $1/2m$ ile başlar. $\beta=\epsilon/(1-\epsilon)$ ile doğruları küçültür, oy ağırlığı $\alpha=\ln1/\beta$'dır. Özgün AdaBoost ile aynı kararı verir; eğitim hatası üstel düşer.
- Kaskad: $D=\prod d_i$, $F=\prod f_i$ ($0.904$, $5.9\times10^{-6}$). Negatif pencere başına ortalama yaklaşık 10 öznitelik hesaplanır. Aşamalar zor negatif madenciliğiyle eğitilir.
- Tarama 640 × 480'de yaklaşık 1.3 milyon pencere ister. `detectMultiScale` parametreleri kapsama ile yanlış pozitif arasında ayar yapar. Yöntem önden, dik yüzlerle sınırlıdır ve bugün CNN'ler daha doğrudur.
