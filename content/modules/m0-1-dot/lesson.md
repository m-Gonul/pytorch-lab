### Vektör: üç bakış açısı {#vektor}

Aynı nesneye üç farklı açıdan bakmak, bu modülün her yerinde işe yarar:

1. **Fizikçinin oku:** yönü ve uzunluğu olan bir ok. Nereden başladığı önemsizdir; aynı yön ve uzunluktaki iki ok aynı vektördür.
2. **Programcının listesi:** sıralı sayılar, $a=(a_1,a_2,\dots,a_d)$. $d$ vektörün **boyutu**dur (PyTorch'ta `a.shape == (d,)`). Bileşenlerin sırası anlamlıdır: $(3,2)\neq(2,3)$.
3. **Veri bilimcinin noktası:** bir örneğin öznitelikleri. Bir ev $(\text{metrekare},\ \text{oda sayısı},\ \text{yaş})$; bir görüntü 784 piksel değeri; bir kelime 300 sayılık bir **gömme** (embedding). Bu bakışta vektör, $d$ boyutlu uzayda bir **nokta**dır.

Üç bakışı birbirine bağlayan şey koordinat sistemidir: oku başlangıcı orijine gelecek şekilde koyarsan ucunun koordinatları listeyi verir.

<figure class="fig"><svg viewBox="0 0 320 230" role="img" aria-label="(3, 2) vektörü: orijinden başlayan ok ve bileşenleri"><defs><marker id="f1a" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker></defs><g class="mute"><line x1="40" y1="20" x2="40" y2="200"/><line x1="20" y1="180" x2="300" y2="180"/><line x1="80" y1="176" x2="80" y2="184"/><line x1="120" y1="176" x2="120" y2="184"/><line x1="160" y1="176" x2="160" y2="184"/><line x1="36" y1="140" x2="44" y2="140"/><line x1="36" y1="100" x2="44" y2="100"/></g><text class="sm" x="80" y="197" text-anchor="middle">1</text><text class="sm" x="120" y="197" text-anchor="middle">2</text><text class="sm" x="160" y="197" text-anchor="middle">3</text><text class="sm" x="30" y="144" text-anchor="end">1</text><text class="sm" x="30" y="104" text-anchor="end">2</text><line x1="40" y1="180" x2="160" y2="180" class="cb dash w2"/><line x1="160" y1="180" x2="160" y2="100" class="cp dash w2"/><line x1="40" y1="180" x2="158" y2="101" class="ca w3" marker-end="url(#f1a)"/><text class="ta" x="58" y="118">a = (3, 2)</text><text class="tb" x="100" y="173">a₁ = 3</text><text class="tp" x="167" y="145">a₂ = 2</text><text class="sm" x="290" y="196">x₁</text><text class="sm" x="48" y="28">x₂</text></svg><figcaption>Şekil 1 — a = (3, 2) vektörü. Ok orijinden (3, 2) noktasına gider; bileşenler okun eksenler boyunca "ne kadar gittiğini" söyler. Turuncu ve yeşil kesikli çizgiler bileşenlerdir.</figcaption></figure>

:::tanım Gösterim
Bu sitede vektörler küçük harfle ($a$, $u$, $w$), bileşenleri alt indisle ($a_i$) yazılır. Matematikte vektörler varsayılan olarak **sütun** vektörüdür; $a^\top$ satır vektörüdür. NumPy/PyTorch'ta 1B dizide satır/sütun ayrımı yoktur; fark ancak 2B'ye (`a[:, None]` sütun, `a[None, :]` satır) geçince ortaya çıkar.
:::

### Vektör işlemleri: toplama, ölçekleme, doğrusal birleşim {#islemler}

**Toplama** bileşen bileşen yapılır: $a+b=(a_1+b_1,\dots,a_d+b_d)$. Geometride: $b$'yi $a$'nın ucuna taşı; toplam, orijinden yeni uca giden oktur (uç uca kuralı). İki sırayla da aynı noktaya varılır — paralelkenarın iki yolu; bu, toplamanın değişme özelliğinin ($a+b=b+a$) resmidir.

**Skalerle çarpma:** $c\,a=(c\,a_1,\dots,c\,a_d)$. $c>0$ yönü korur, uzunluğu $|c|$ katına çıkarır; $c<0$ yönü ters çevirir; $c=0$ sıfır vektörünü verir.

**Doğrusal birleşim:** $c_1a+c_2b$. Bir sinir ağı katmanının yaptığı şey tam olarak budur: çıktı, ağırlık matrisinin sütunlarının girdi bileşenleriyle ağırlıklandırılmış doğrusal birleşimidir (M0.2).

<figure class="fig"><svg viewBox="0 0 320 240" role="img" aria-label="Paralelkenar kuralı: a, b ve a + b"><defs><marker id="f2a" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker><marker id="f2b" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker><marker id="f2p" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cp-f"/></marker></defs><g class="mute"><line x1="40" y1="20" x2="40" y2="215"/><line x1="20" y1="200" x2="305" y2="200"/></g><polygon points="40,200 160,160 200,80 80,120" class="soft-p"/><line x1="160" y1="160" x2="199" y2="82" class="cb dash"/><line x1="80" y1="120" x2="198" y2="81" class="ca dash"/><line x1="40" y1="200" x2="158" y2="161" class="ca w3" marker-end="url(#f2a)"/><line x1="40" y1="200" x2="79" y2="122" class="cb w3" marker-end="url(#f2b)"/><line x1="40" y1="200" x2="198" y2="82" class="cp w3" marker-end="url(#f2p)"/><text class="ta" x="110" y="196">a = (3, 1)</text><text class="tb" x="16" y="150">b = (1, 2)</text><text class="tp" x="205" y="76">a + b = (4, 3)</text></svg><figcaption>Şekil 2 — Paralelkenar kuralı. Önce a sonra b (alt-sağ yol) ya da önce b sonra a (üst-sol yol): ikisi de (4, 3)'e varır. Kesikli oklar taşınmış kopyalardır.</figcaption></figure>

### İç çarpımın cebirsel tanımı ve özellikleri {#tanim}

:::tanım İç çarpım (dot product, nokta çarpım)
Aynı boyutlu iki gerçel vektör için
$$a\cdot b=a^\top b=\sum_{i=1}^{d}a_ib_i=a_1b_1+a_2b_2+\dots+a_db_d .$$
Sonuç bir **skaler**dir (tek sayı). Boyutlar farklıysa tanımsızdır: PyTorch `torch.dot` hata verir.
:::

**Örnek:** $a=(1,2,3)$, $b=(4,-5,6)$: $a\cdot b=4-10+18=12$. Negatif bir terim toplamı düşürür; işaret, aşağıda göreceğimiz gibi açı hakkında bilgi taşır.

İç çarpımı "iç çarpım" yapan üç özellik vardır (her gerçel vektör uzayında iç çarpımın aksiyomları bunlardır):

1. **Simetri:** $a\cdot b=b\cdot a$ — çünkü $a_ib_i=b_ia_i$.
2. **Doğrusallık:** $(c\,a+d\,a')\cdot b=c\,(a\cdot b)+d\,(a'\cdot b)$ — toplamın her terimi doğrusal. Bu yüzden iç çarpım toplamın üzerine "dağılır": $(a+a')\cdot b=a\cdot b+a'\cdot b$.
3. **Pozitif tanımlılık:** $a\cdot a=\sum a_i^2\ge0$ ve eşitlik **ancak** $a=0$ iken — karelerin toplamı ancak hepsi sıfırsa sıfırdır.

:::uyarı İç çarpım birleşme özelliğine sahip değildir
$(a\cdot b)\cdot c$ bir skalerin bir vektörle çarpımıdır (vektör), $a\cdot(b\cdot c)$ ise başka bir vektördür; genelde eşit değildir. "Üç vektörün iç çarpımı" diye bir şey yoktur.
:::

### Geometrik anlam: ‖a‖‖b‖ cos θ {#geometri}

Cebirsel tanım hesaplamak için, geometrik tanım **anlamak** için:
$$a\cdot b=\|a\|\,\|b\|\cos\theta ,$$
burada $\theta\in[0^\circ,180^\circ]$ iki ok arasındaki açı, $\|a\|=\sqrt{a\cdot a}$ uzunluktur.

<figure class="fig"><svg viewBox="0 0 320 220" role="img" aria-label="Kosinüs teoremi üçgeni: a, b ve a − b kenarları, aradaki θ açısı"><defs><marker id="f3a" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker><marker id="f3b" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker><marker id="f3q" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cq-f"/></marker></defs><line x1="50" y1="190" x2="268" y2="171" class="ca w3" marker-end="url(#f3a)"/><line x1="50" y1="190" x2="139" y2="62" class="cb w3" marker-end="url(#f3b)"/><line x1="140" y1="60" x2="266" y2="168" class="cq w2 dash" marker-end="url(#f3q)"/><path d="M 95 186 A 45 45 0 0 0 76 153" class="ink"/><text x="95" y="168">θ</text><text class="ta" x="160" y="200">a</text><text class="tb" x="72" y="115">b</text><text class="tq" x="210" y="105">a − b</text></svg><figcaption>Şekil 3 — a, b ve aralarındaki fark a − b bir üçgen oluşturur. Kosinüs teoremi bu üçgenin üçüncü kenarını θ ile ilişkilendirir; iç çarpımın geometrik formülü buradan çıkar.</figcaption></figure>

:::türetim Kosinüs teoreminden a·b = ‖a‖‖b‖cos θ
**İddia:** sıfırdan farklı $a,b$ için $a\cdot b=\|a\|\|b\|\cos\theta$. **Kullanılanlar:** iç çarpımın doğrusallığı ve simetrisi, düzlem geometrisindeki kosinüs teoremi ($c^2=x^2+y^2-2xy\cos\gamma$).

1. Şekil 3'teki üçgenin kenarları $\|a\|$, $\|b\|$ ve $\|a-b\|$; $\|a\|$ ile $\|b\|$ arasındaki açı $\theta$. Kosinüs teoremi: $\|a-b\|^2=\|a\|^2+\|b\|^2-2\|a\|\|b\|\cos\theta$. *(geometri)*
2. Aynı uzunluğu iç çarpımla aç: $\|a-b\|^2=(a-b)\cdot(a-b)=a\cdot a-a\cdot b-b\cdot a+b\cdot b=\|a\|^2+\|b\|^2-2\,a\cdot b$. *(doğrusallık, sonra simetri: $b\cdot a=a\cdot b$)*
3. 1 ve 2 aynı niceliği veriyor; eşitle ve $\|a\|^2+\|b\|^2$'yi iki yandan at: $-2\,a\cdot b=-2\|a\|\|b\|\cos\theta$. ∎

**Ne söyler:** iç çarpımın işareti açının 90°'den küçük mü büyük mü olduğunu söyler; $a\cdot b=0\iff$ vektörler **dik** (ortogonal). **Ne söylemez:** $d>3$ boyutta "açı"yı gözümüzde canlandıramayız; orada bu formül açının **tanımı** olur: $\cos\theta:=\frac{a\cdot b}{\|a\|\|b\|}$. Bunun her zaman $[-1,1]$'de kaldığını Cauchy–Schwarz garanti eder (aşağıda).
:::

:::sezgi "Ne kadar aynı yöne bakıyorlar?"
$\cos\theta$ yönlerin uyumunu ölçer (+1 aynı yön, 0 dik, −1 zıt); $\|a\|\|b\|$ ise büyüklükleri. İç çarpım ikisini çarpar: uzun ve aynı yöne bakan vektörlerin iç çarpımı büyüktür. Aşağıdaki şekilde uçları sürükle ve tablodaki iki hesabın (bileşenlerden ve uzunluk–açıdan) her zaman aynı sayıyı verdiğini gör.
:::

::lab l1

### İzdüşüm ve dik ayrışım {#izdusum}

$a$'nın $b$ yönündeki **gölgesi** ne kadar? $b$ yönündeki birim vektör $\hat b=b/\|b\|$. Skaler izdüşüm $a\cdot\hat b=\|a\|\cos\theta$ (gölgenin işaretli uzunluğu), vektör izdüşüm
$$\mathrm{proj}_b\,a=(a\cdot\hat b)\,\hat b=\frac{a\cdot b}{b\cdot b}\,b .$$

<figure class="fig"><svg viewBox="0 0 320 220" role="img" aria-label="a vektörünün b üzerine izdüşümü ve dik artık"><defs><marker id="f4a" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker><marker id="f4b" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker><marker id="f4p" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cp-f"/></marker></defs><line x1="40" y1="190" x2="288" y2="150" class="cb w2" marker-end="url(#f4b)"/><line x1="40" y1="190" x2="185" y2="167" class="cp w3" marker-end="url(#f4p)" opacity="0.9"/><line x1="40" y1="190" x2="168" y2="62" class="ca w3" marker-end="url(#f4a)"/><line x1="170" y1="60" x2="187" y2="166" class="ink dash"/><polyline points="176,168 174,156 186,154" class="ink"/><text class="ta" x="90" y="110">a</text><text class="tb" x="270" y="140">b</text><text class="tp" x="80" y="208">proj<tspan baseline-shift="-3" font-size="75%">b</tspan> a</text><text class="sm" x="182" y="110">a − proj<tspan baseline-shift="-3" font-size="75%">b</tspan> a</text><text class="sm" x="196" y="122">(b'ye dik)</text></svg><figcaption>Şekil 4 — a'nın b üzerine izdüşümü (yeşil) ve artık a − proj (kesikli). Artık b'ye diktir; a, "b boyunca" ve "b'ye dik" iki parçaya ayrılır.</figcaption></figure>

:::türetim Artık neden diktir?
$r=a-\frac{a\cdot b}{b\cdot b}b$ olsun. $r\cdot b=a\cdot b-\frac{a\cdot b}{b\cdot b}(b\cdot b)=0$. Dolayısıyla her $a$ **tek bir biçimde** $a=\underbrace{\mathrm{proj}_ba}_{b\text{ boyunca}}+\underbrace{r}_{\perp b}$ olarak yazılır ve Pisagor: $\|a\|^2=\|\mathrm{proj}_ba\|^2+\|r\|^2$ (çapraz terim $2\,\mathrm{proj}\cdot r=0$).

**En iyi yaklaşım:** $b$ doğrultusundaki tüm vektörler içinde $a$'ya en yakını izdüşümdür: $\|a-tb\|^2$'yi $t$'ye göre türetip sıfırla: $-2b\cdot(a-tb)=0\Rightarrow t=\frac{a\cdot b}{b\cdot b}$. Bu, **en küçük kareler**in tek boyutlu hâlidir; M1.1'de $b$ yerine $X$'in sütun uzayı gelir ve aynı diklik koşulu normal denklemleri verir.
:::

Deney alanında görünümü "İzdüşüm" seçip $a$'yı $b$'ye dik konuma getir: yeşil izdüşüm sıfıra iner. $a$'yı $b$'nin arkasına (açı > 90°) taşı: izdüşüm $b$'nin **ters** yönüne döner — skaler izdüşüm negatiftir.

### Norm: bir vektörün "büyüklüğü" {#norm}

**Öklid normu (L2):** $\|a\|_2=\sqrt{a\cdot a}=\sqrt{\sum a_i^2}$ — Pisagor teoreminin $d$ boyuta genellemesi. $(3,4)$'ün normu 5; $(1,2,3)$'ün $\sqrt{14}\approx3.742$.

:::tanım Norm aksiyomları
Bir $\|\cdot\|$ fonksiyonu şu üç koşulu sağlıyorsa normdur:
1. $\|a\|\ge0$, eşitlik ancak $a=0$ iken (pozitiflik),
2. $\|c\,a\|=|c|\,\|a\|$ (mutlak homojenlik — iki katı uzun ok, iki kat norm),
3. $\|a+b\|\le\|a\|+\|b\|$ (üçgen eşitsizliği — "kestirme yol daha kısadır").
:::

Öklid normu tek seçenek değildir. **Lp normları:** $\|a\|_p=\big(\sum_i|a_i|^p\big)^{1/p}$, $p\ge1$:

| norm | formül | ML'de |
|---|---|---|
| L1 (Manhattan) | $\sum\lvert a_i\rvert$ | lasso cezası (seyreklik), MAE kaybı |
| L2 (Öklid) | $\sqrt{\sum a_i^2}$ | ridge/ağırlık bozunması, MSE, gradyan kırpma |
| L∞ (Chebyshev) | $\max_i\lvert a_i\rvert$ | adversaryal saldırılarda bozulma bütçesi |

Hangi vektörlerin "uzunluğu 1" olduğu (birim top) norma göre bambaşka bir şekildir: L1'de köşeleri eksenlerde olan bir elmas, L2'de çember, L∞'da kare. Aşağıda $p$'yi değiştirerek dönüşümü izle; $p<1$'de neyin bozulduğunu da gör.

::lab l2

**Normalizasyon:** $\hat a=a/\|a\|$ birim vektördür (yalnız yön). Sıfır vektörü normalize edilemez; uygulamada $a/(\|a\|+\varepsilon)$ yazılır (`F.normalize` bunu yapar).

### Cauchy–Schwarz ve üçgen eşitsizliği {#cs}

:::kanıt Cauchy–Schwarz: |a·b| ≤ ‖a‖ ‖b‖
**Koşul:** gerçel vektörler, herhangi bir boyut. **Kanıt:** $b=0$ ise iki taraf 0. Değilse her gerçel $t$ için uzunluğun karesi negatif olamaz:
$$0\le\|a-tb\|^2=\|b\|^2t^2-2(a\cdot b)\,t+\|a\|^2 .$$
Sağ taraf $t$'nin hiç negatif olmayan ikinci dereceden bir polinomu; böyle bir polinomun iki farklı gerçel kökü olamaz, yani diskriminantı ≤ 0: $4(a\cdot b)^2-4\|a\|^2\|b\|^2\le0$. Karekök alınca iddia. **Eşitlik** ancak polinomun bir kökü varsa, yani bir $t$ için $a=tb$ (paralel vektörler). ∎

**Sonuç:** $-1\le\frac{a\cdot b}{\|a\|\|b\|}\le1$: yüksek boyutta "açı" tanımı her zaman anlamlıdır.
:::

:::kanıt Üçgen eşitsizliği: ‖a + b‖ ≤ ‖a‖ + ‖b‖
$\|a+b\|^2=\|a\|^2+2\,a\cdot b+\|b\|^2\le\|a\|^2+2\|a\|\|b\|+\|b\|^2=(\|a\|+\|b\|)^2$. Ortadaki eşitsizlik Cauchy–Schwarz'dır ($a\cdot b\le|a\cdot b|\le\|a\|\|b\|$). Kök alınca iddia; eşitlik ancak $a$ ve $b$ aynı yöndeyse. ∎ Yani Öklid normu gerçekten bir normdur.
:::

### Kosinüs benzerliği, Öklid uzaklığı ve aralarındaki bağ {#benzerlik}

İki gömme vektörü ne kadar "benzer"? İki yaygın ölçü:

- **Kosinüs benzerliği** $\cos(a,b)=\frac{a\cdot b}{\|a\|\|b\|}\in[-1,1]$ — yalnız **yöne** bakar; uzunluk önemsizdir.
- **Öklid uzaklığı** $\|a-b\|$ — yöne de büyüklüğe de bakar.

İkisini birleştiren özdeşlik (Şekil 3'ün ikinci adımı): $\|a-b\|^2=\|a\|^2+\|b\|^2-2\,a\cdot b$. Vektörler **birim normlu** ise
$$\|a-b\|^2=2-2\cos(a,b),$$
yani en yakın komşuyu Öklid'e göre ya da kosinüse göre aramak **aynı sıralamayı** verir. Bu yüzden gömme aramasında vektörler önce normalize edilir ve sonra hızlı iç çarpımla aranır.

:::örnek Belgede kelime sayıları
$a=(2,1,0)$ ("kedi" 2 kez, "süt" 1 kez), $b=(4,2,0)$ (aynı belge iki kez yazılmış), $c=(0,1,2)$ (başka konu). $\cos(a,b)=1$ (aynı yön), $\|a-b\|=\sqrt5\approx2.24$; $\cos(a,c)=\frac15=0.2$, $\|a-c\|=\sqrt{8}\approx2.83$. Öklid'e göre $b$ ile $c$ neredeyse eşit uzaklıkta; kosinüse göre $b$ açıkça aynı konu. Uzunluk (belge uzunluğu) bilgisi istenmiyorsa kosinüs doğru seçimdir (M2.1 P7).
:::

:::yanılgı "Kosinüs benzerliği 0 ise vektörler ilgisizdir"
Yalnız **doğrusal** ilişki yoktur demektir. Bileşenleri hepsi pozitif vektörlerde (ör. ReLU çıktıları) kosinüs hiçbir zaman negatif olmaz ve rastgele çiftler bile ~0.75 çıkar (aşağıdaki deneyde "Düzgün [0, 1]" seçeneği). Karşılaştırmadan önce ortalamayı çıkarmak (merkezleme) kosinüsü **korelasyona** dönüştürür.
:::

### Makine öğrenmesinde iç çarpım {#ml}

**Bir nöron:** $z=w\cdot x+b$. Ağırlık vektörü $w$ "aradığı deseni" temsil eder; girdi bu desenle aynı yöne baktıkça $z$ büyür. Bu, bir **şablon eşleme** (template matching) işlemidir.

**Karar sınırı:** $w\cdot x+b=0$ denklemini sağlayan noktalar bir **hiperdüzlem**dir (2B'de doğru, 3B'de düzlem). $w$ bu düzleme **diktir** (normal vektör): düzlem üzerindeki iki nokta için $w\cdot(x_1-x_2)=0$. Bir noktanın düzleme işaretli uzaklığı $\frac{w\cdot x+b}{\|w\|}$ — SVM'nin marjı buradan gelir (M2.3).

<figure class="fig"><svg viewBox="0 0 320 240" role="img" aria-label="Hiperdüzlem w·x + b = 0, normal vektör w, iki taraftaki noktalar ve bir noktanın uzaklığı"><defs><marker id="f5w" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><polygon points="40,220 280,40 320,40 320,240 40,240" class="soft-b"/><line x1="40" y1="220" x2="280" y2="40" class="cb w2"/><line x1="160" y1="130" x2="195" y2="176" class="cb w3" marker-end="url(#f5w)"/><text class="tb" x="200" y="190">w</text><text class="sm" x="228" y="52">w·x + b = 0</text><circle cx="230" cy="170" r="5" class="cp-f"/><circle cx="250" cy="120" r="5" class="cp-f"/><circle cx="285" cy="200" r="5" class="cp-f"/><circle cx="80" cy="90" r="5" class="cq-f"/><circle cx="120" cy="60" r="5" class="cq-f"/><circle cx="60" cy="150" r="5" class="cq-f"/><line x1="250" y1="120" x2="222" y2="83" class="ink dash"/><text class="sm" x="246" y="96">uzaklık = (w·x + b)/‖w‖</text><text class="tp" x="255" y="228">w·x + b > 0</text><text class="tq" x="50" y="40">w·x + b &lt; 0</text></svg><figcaption>Şekil 5 — Doğrusal bir sınıflandırıcının geometrisi. w karar sınırına diktir; bir noktanın tarafını w·x + b'nin işareti, sınıra uzaklığını (w·x + b)/‖w‖ belirler. Deney alanında görünümü "İşaret bölgeleri" yap: b'yi döndürdükçe sınır da döner.</figcaption></figure>

**Attention:** sorgu $q$ ile anahtar $k$'nın uyumu $q\cdot k$ ile ölçülür, $\sqrt{d}$'ye bölünür ve softmax'a girer (M4.6). **Gömme araması:** milyonlarca vektör içinde bir sorguya en yakınları bulmak = en büyük iç çarpımı (MIPS) bulmak. **Gradyan:** $\nabla_w(w\cdot x)=x$ — doğrusal bir katmanda ağırlığın gradyanı girdinin kendisidir (M3.2).

### Yüksek boyutta sezgi neden yanıltır? {#yuksek-boyut}

2B ve 3B'deki sezgiyle "rastgele iki ok rastgele bir açı yapar" deriz. $d=1000$'de durum farklıdır: bağımsız, sıfır ortalamalı rastgele vektörler **neredeyse her zaman neredeyse diktir**. Nedeni: $a\cdot b=\sum_i a_ib_i$, $d$ tane bağımsız sıfır ortalamalı terimin toplamı — std'si $\sqrt d$; oysa $\|a\|\|b\|\approx d$. Oran $1/\sqrt d$ ölçeğinde 0 civarına yığılır.

::lab l3

Bunun iki pratik sonucu var: (1) yüksek boyutlu gömmelerde 0.2'lik bir kosinüs bile rastlantı değil, anlamlı bir benzerlik olabilir; (2) attention'da skorları $\sqrt{d}$'ye bölmezsek softmax'a giren sayıların ölçeği boyutla büyür (M4.6).

### Hesaplama: maliyet, sayısal kararlılık, vektörleştirme {#hesap}

**Maliyet:** $d$ çarpma + $d-1$ toplama: $O(d)$. $n$ vektörün tüm çiftleri arasındaki benzerlik $O(n^2d)$ — ama tek bir matris çarpımıdır: satırları normalize edilmiş $X$ (n × d) için $S=XX^\top$ (c ve e aşamaları).

**Döngü yerine vektörleştirme:** Python döngüsü eleman başına yorumlayıcı yükü ödetir; `np.dot`/`@` aynı işi derlenmiş, SIMD kullanan koda devreder (e aşamasında hızı ölçeceksin). **`einsum`** indis notasyonuyla yazar: `"i,i->"` iç çarpım, `"nd,nd->n"` satır satır iç çarpımlar, `"nd,md->nm"` tüm çiftler. Kural: çıktıda geçmeyen indisler toplanır.

:::uyarı Sayısal kararlılık
- **Kayan nokta toplamı birleşmeli değildir:** $(10^{8}+1)-10^{8}$ float32'de 0'dır. Uzun iç çarpımlarda toplama sırası sonucu (son basamaklarda) değiştirir; GPU'da paralel indirgeme bu yüzden CPU sonucundan biraz farklı çıkar. Testlerde `==` yerine tolerans kullan.
- **Norm hesabında taşma:** $\sqrt{\sum a_i^2}$, bileşenler ~$10^{20}$ ise float32'de karede taşar (inf). Kararlı yol: en büyük bileşene böl, normu al, geri çarp (`np.linalg.norm` ve `torch.linalg.norm` bunu dikkatle yapar; `math.hypot` da).
- **Sıfıra bölme:** kosinüste paydaya küçük bir $\varepsilon$ ekle.
:::

### Özet ve sembol tablosu {#ozet}

| sembol | anlamı | birim / tür |
|---|---|---|
| $a,b\in\mathbb R^d$ | $d$ boyutlu gerçel vektörler | sayı listesi |
| $a_i$ | $i$. bileşen | skaler |
| $a\cdot b=a^\top b$ | iç çarpım $\sum a_ib_i$ | skaler (birimi bileşen birimlerinin çarpımı) |
| $\lVert a\rVert$ | Öklid normu $\sqrt{a\cdot a}$ | bileşenlerle aynı birim |
| $\theta$ | iki vektör arasındaki açı | derece / radyan, $[0,\pi]$ |
| $\hat a$ | birim vektör $a/\lVert a\rVert$ | birimsiz |
| $\mathrm{proj}_ba$ | $a$'nın $b$ üzerine izdüşümü | vektör |

**Akılda kalsın:** iç çarpım = bileşenlerin çarpımlarının toplamı = uzunluklar × açının kosinüsü. İşareti yön uyumunu, sıfırı dikliği söyler; izdüşüm, en küçük kareler ve doğrusal sınıflandırıcılar bu tek fikrin üzerine kurulur. Sıradaki modül (matris çarpımı), çok sayıda iç çarpımı aynı anda yapmanın dilidir: her çıktı elemanı bir satır ile bir sütunun iç çarpımıdır.

**Kaynaklar ve daha fazlası:** Strang, *Introduction to Linear Algebra*, Bölüm 1.2 (uzunluklar ve iç çarpımlar); Deisenroth, Faisal, Ong, *Mathematics for Machine Learning*, Bölüm 3.1–3.4 (normlar, iç çarpımlar, açılar, izdüşümler); 3Blue1Brown, *Essence of Linear Algebra* 9. bölüm (iç çarpım ve dualite); d2l.ai §2.3.10.
