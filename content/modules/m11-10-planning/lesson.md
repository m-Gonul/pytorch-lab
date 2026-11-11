### Modeli öğren, planla: neden modelli? {#neden-modelli}

M11.4–M11.9'daki yöntemlerin hepsi **modelsizdi** (model-free): eylem → gözlem → ödül döngüsünden gelen örneklerle değeri ya da politikayı doğrudan güncellediler; geçiş ve ödül modelini hiç öğrenmediler. Haydutlarda (M11.7–M11.8) yalnız ödül modeli vardı. **Modelli** (model-based) RL bu eksik parçayı ekler: ortamın dinamiğini ve ödülünü açık bir model olarak öğrenir ve bu modeli kullanarak **planlar** — gerçek ortamda yaşanmadan önce zihinsel deney yapar.

Kazanç ve bedel:

- **Kazanç: örnek verimliliği.** Gerçek her adım, model üzerinden üretilen çok sayıda simüle adıma dönüştürülebilir. Gerçek deneyi pahalı olan problemlerde (robot, süreç kontrolü) bu kritiktir.
- **Bedel: karmaşıklık ve model hatası.** Ek bir öğrenme problemi doğar; model yanlışsa plan da yanlıştır. Sinir ağı modelinde küçük hatalar çok adımlı ön bakışta **birikir** (aşağıda ölçülecek).

Modelli ajanın döngüsü her adımda dört iş yapar: eylem seç ve uygula → $s'$ ve $r$ gözle → modeli güncelle → modelle planla (değeri ya da politikayı güncelle). Modelliğin iki ekseni vardır: **model nasıl öğrenilir** (tablo, doğrusal, sinir ağı) ve **plan nasıl yapılır** (tam değer yinelemesi, birkaç adımlık kısmi planlama, şimdiki durumdan ileri arama).

### Tablo model ve değer yinelemesiyle modelli RL {#model-vi}

Sonlu bir MDP'de model bir tablodur: her $(s,a)$ çifti için geçiş dağılımı $\hat P(s'\mid s,a)$ ve beklenen ödül $\hat R(s,a)$. En doğal kestirim **göreli sayımlardır**:

$$\hat P(s'\mid s,a)=\frac{n(s,a,s')}{n(s,a)},\qquad \hat R(s,a)=\frac1{n(s,a)}\sum_k r_k.$$

:::türetim Sayım kestirimi en çok olabilirlik kestirimidir
Gözlemler bağımsız ve aynı dağılımlıysa, $n(s,a)$ denemede $s'$e $n(s,a,s')$ kez gitmiş olmanın olasılığı çok terimlidir (multinomial):

$$L=\prod_{s'}P(s'\mid s,a)^{\,n(s,a,s')}.$$

$\log L=\sum_{s'}n(s,a,s')\log P(s'\mid s,a)$ toplamı $\sum_{s'}P(s'\mid s,a)=1$ kısıtı altında en çoklanır. Lagrange çarpanıyla türev sıfırlanınca $P(s'\mid s,a)\propto n(s,a,s')$ çıkar; normalize edilince $\hat P=n(s,a,s')/n(s,a)$. Aynı argüman ödülde Gauss ya da kategorik olasılık varsayımıyla gözlenen ödüllerin ortalamasını verir. Az veride $\hat P$ sıfırlara yaslanır; iyimserlik ya da önce (prior) eklenebilir (M11.8'in Bayesçi doğrusal regresyonuyla aynı fikir).
:::

Ödül ortalaması artımlı da yazılır: $n.$ gözlemde $\hat R\leftarrow\bigl(r+(n-1)\hat R\bigr)/n$ — M11.4'teki artımlı ortalamanın aynısı. Modelli RL'in en yalın hâli her gerçek adımdan sonra modeli güncelleyip **değer yinelemesini (M11.2) sıfırdan çözmektir**: $V^*(s)=\max_a\bigl[\hat R(s,a)+\gamma\sum_{s'}\hat P(s'\mid s,a)V^*(s')\bigr]$. Bu, **kesinlik eşdeğerli** (certainty equivalence) plandır: kestirilen modeli gerçek bilinmek yerine koyar. c aşaması tam olarak bunu NumPy'da yapıyor: sayım modeli + değer yinelemesi, ve aynı veri üzerinde tablo Q-öğrenmeyle karşılaştırma.

Sorun ölçekte: $|S|$ durum için bir değer yinelemesi turu $O(|S|^2|A|)$ işlem ister ve bunu her gerçek adımda yapmak olanaksızdır. Slayt 9'un çözümü **kısmi planlamadır**: her adımda birkaç Q-öğrenme (ya da politika gradyanı) adımı at.

### Sayılamayacak durumlar: model yaklaşımı {#model-yaklasimi}

Durum uzayı büyük ya da sürekliyse tablo tutulamaz; model bir fonksiyon yaklaşımı olur:

- **Doğrusal model:** $p(s'\mid s,a)=\mathcal N(s'\mid w^\top x(s,a),\ \sigma^2I)$. Az parametre, analitik; karmaşık dinamikte yetersiz.
- **Gauss süreci:** belirsizliği de taşır (nerede veri yoksa orada varyans büyür); ölçeklenmesi zordur.
- **Sinir ağı:** genelde deterministik $\hat s'=s+\Delta(s,a)$ tahmin edilir; artık (delta) tahmini öğrenmeyi kolaylaştırır (küçük değerler, sıfır civarı hedef). d aşaması tam böyle bir modelle CartPole'da **model öngörülü kontrol** (MPC) yapıyor.

İki uyarı: (1) Sinir ağı modelleri deterministik diye sınıflamak eksiktir — çıkışı bir dağılım veren (ör. Gauss ortalaması ve varyansı üreten) ağlar stokastik model de öğrenir; belirsizliği ölçmek model hatasını fark etmek için değerlidir. (2) Tek adım hatası küçük bir model bile $H$ adımlı ön bakışta $H$ kat hata biriktirebilir; MPC'de ufuk $H$ bu yüzden kısılıdır ve modellsiz ince ayar (model-free fine-tuning) ile birleştirilir.

### Kısmi planlama: Dyna-Q {#dyna}

**Dyna-Q** (Sutton 1991) modelli ile modelsiz öğrenmeyi tek döngüde birleştirir:

```text
Dyna-Q(s):
    repeat:
        a ← davranış politikası (ε-açgözlü); uygula, s' ve r gözle
        Q-öğrenme adımı:  Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]
        modeli güncelle:  model(s,a) ← (r, s')
        repeat n kez:                          ← planlama
            (ŝ, â) ← gerçekte görülmüş bir (s,a) çifti
            (r̂, ŝ') ← model(ŝ, â)
            Q(ŝ, â) ← Q(ŝ, â) + α[r̂ + γ max_â' Q(ŝ', â') − Q(ŝ, â)]
        s ← s'
```

$n$ planlama adımı, gerçek deneyimi en çok $n+1$ katına çıkarır: her gerçek adım bir gerçek Q güncellemesi ve $n$ simüle güncelleme üretir. Labirentte bu etki ölçülüyor (aşağıda).

:::uyarı Kaynak düzeltmesi (lect_12): planlama çiftleri keyfî değil, görülmüş olmalı
Slayt 10 ve 13, planlamada "$(ŝ,â)$'yı keyfî (arbitrarily) örnekleyin" diyor. Tablo modelde hiç denenmemiş bir $(ŝ,â)$ için model değeri tanımsızdır: sıfır dolgu ya da kendine dönüş varsaymak Q'yu amaçtan uzaklaştırır. Dyna'nın özgün tanımında planlama, **daha önce gerçekten denenmiş** $(s,a)$ çiftlerinden seçim yapar (Sutton ve Barto, Algoritma 8.1); bizim uygulamamız da `keys` listesinde yalnız görülmüş çiftleri tutuyor. Fonksiyon yaklaşımılı modelde ise görülmemiş girdiye genelleme modelin işidir — orada keyfî örneklemek anlamlı olabilir, ama tabloda değil.
:::

<figure class="fig wide"><svg viewBox="0 0 640 286" role="img" aria-label="Dyna labirenti: 6 satır 9 sütun ızgara, duvarlar gri; başlangıç B, hedef G; en kısa yol 14 adımda vurgulu. Sağda ikinci bölümdeki ortalama adım sayısı: n = 0 için 750 civarı, n = 10 için 140 civarı, n = 50 için 100 civarı."><rect x="253" y="35" width="28" height="28" class="mute-f"/><rect x="103" y="65" width="28" height="28" class="mute-f"/><rect x="253" y="65" width="28" height="28" class="mute-f"/><rect x="103" y="95" width="28" height="28" class="mute-f"/><rect x="253" y="95" width="28" height="28" class="mute-f"/><rect x="103" y="125" width="28" height="28" class="mute-f"/><rect x="193" y="155" width="28" height="28" class="mute-f"/><rect x="42" y="34" width="270" height="180" class="mute"/><line x1="42" y1="34" x2="42" y2="214" class="mute"/><line x1="72" y1="34" x2="72" y2="214" class="mute"/><line x1="102" y1="34" x2="102" y2="214" class="mute"/><line x1="132" y1="34" x2="132" y2="214" class="mute"/><line x1="162" y1="34" x2="162" y2="214" class="mute"/><line x1="192" y1="34" x2="192" y2="214" class="mute"/><line x1="222" y1="34" x2="222" y2="214" class="mute"/><line x1="252" y1="34" x2="252" y2="214" class="mute"/><line x1="282" y1="34" x2="282" y2="214" class="mute"/><line x1="312" y1="34" x2="312" y2="214" class="mute"/><line x1="42" y1="34" x2="312" y2="34" class="mute"/><line x1="42" y1="64" x2="312" y2="64" class="mute"/><line x1="42" y1="94" x2="312" y2="94" class="mute"/><line x1="42" y1="124" x2="312" y2="124" class="mute"/><line x1="42" y1="154" x2="312" y2="154" class="mute"/><line x1="42" y1="184" x2="312" y2="184" class="mute"/><line x1="42" y1="214" x2="312" y2="214" class="mute"/><rect x="43" y="95" width="28" height="28" class="soft-a"/><text x="57.0" y="113.5" class="sm" text-anchor="middle">B</text><rect x="283" y="35" width="28" height="28" class="soft-b"/><text x="297.0" y="53.5" class="sm" text-anchor="middle">G</text><polyline points="57.0,109.0 57.0,139.0 57.0,169.0 87.0,169.0 117.0,169.0 147.0,169.0 147.0,139.0 177.0,139.0 207.0,139.0 237.0,139.0 267.0,139.0 297.0,139.0 297.0,109.0 297.0,79.0 297.0,49.0" class="ca w3" fill="none"/><circle cx="57.0" cy="109.0" r="3.4" class="ca-f"/><circle cx="57.0" cy="139.0" r="3.4" class="ca-f"/><circle cx="57.0" cy="169.0" r="3.4" class="ca-f"/><circle cx="87.0" cy="169.0" r="3.4" class="ca-f"/><circle cx="117.0" cy="169.0" r="3.4" class="ca-f"/><circle cx="147.0" cy="169.0" r="3.4" class="ca-f"/><circle cx="147.0" cy="139.0" r="3.4" class="ca-f"/><circle cx="177.0" cy="139.0" r="3.4" class="ca-f"/><circle cx="207.0" cy="139.0" r="3.4" class="ca-f"/><circle cx="237.0" cy="139.0" r="3.4" class="ca-f"/><circle cx="267.0" cy="139.0" r="3.4" class="ca-f"/><circle cx="297.0" cy="139.0" r="3.4" class="ca-f"/><circle cx="297.0" cy="109.0" r="3.4" class="ca-f"/><circle cx="297.0" cy="79.0" r="3.4" class="ca-f"/><circle cx="297.0" cy="49.0" r="3.4" class="ca-f"/><text x="177.0" y="232.0" class="sm" text-anchor="middle">en kısa yol: 14 adım (BFS)</text><text x="177.0" y="24.0" class="sm" text-anchor="middle">Sutton–Barto labirenti (6 × 9)</text><text x="485.0" y="24.0" class="sm" text-anchor="middle">ikinci bölümde ortalama adım</text><rect x="400" y="86.0" width="52" height="150.0" class="cb-f"/><text x="426.0" y="80.0" class="sm" text-anchor="middle">752</text><text x="426.0" y="251.0" class="sm" text-anchor="middle">n = 0</text><rect x="478" y="171.3" width="52" height="64.7" class="ca-f"/><text x="504.0" y="165.3" class="sm" text-anchor="middle">140</text><text x="504.0" y="251.0" class="sm" text-anchor="middle">n = 10</text><rect x="556" y="181.9" width="52" height="54.1" class="ca-f"/><text x="582.0" y="175.9" class="sm" text-anchor="middle">97.8</text><text x="582.0" y="251.0" class="sm" text-anchor="middle">n = 50</text><line x1="390" y1="236" x2="602" y2="236" class="mute"/><text x="465.0" y="268.0" class="sm" text-anchor="middle">planlama adımı n (30 çalıştırma)</text></svg>
<figcaption>Şekil 1 — Sutton–Barto labirenti (Şekil 8.2): başlangıç B, hedef G, gri hücreler duvar. Kırmızı yol BFS ile bulunan en kısa yol (14 adım). Sağdaki çubuklar ikinci bölümde ortalama adım sayısı: n = 0 için 752, n = 10 için 140, n = 50 için 97.8 (30 çalıştırma ortalaması; çubuk yükseklikleri karekök ölçekli). Bütün değerler betikte hesaplandı.</figcaption></figure>


Şekil 1'deki ölçüm tablosu net bir tablo çizer: ikinci bölümde ortalama adım sayısı 30 çalıştırmada n = 0 için 752, n = 10 için 140, n = 50 için 98. Planlamanın getirisi ilk bölümlerde devasadır (4–8 kat kısalma); ama dağılım ağır kuyrukludur: n = 50'de bile 8 tohumda ikinci bölüm 37 ile 212 arasında değişti. Sebebi keşfin bağımlılığıdır: bölüm 1'de yanlış yöne açılan bir rastgele yürüyüş, Q'yu hedefe götüren geçişleri geç öğrenebilir. Aşağıdaki deney bunu canlı gösteriyor.

::lab l1

**Dyna-Q ve tekrar tamponu.** Dyna-Q'nun planlama adımı, DQN'in tekrar tamponunun (M11.6) yaptığı işe yapısal olarak çok benzer: geçmiş örneklerle Q günceller. Fark ürettiği örneklerdedir.

:::uyarı Kaynak düzeltmesi (lect_12): "tampon genelleme yapmaz" ifadesi nüans ister
Slayt 11, tekrar tamponu için "basit, gerçek örnekler, başka durum–eylem çiftlerine genelleme yok" diyor. Bu **tablo Q** için doğrudur: tampon $(s,a)$ çiftlerini birebir tekrar oynatır. Ama Q bir sinir ağıysa (DQN'deki gibi), tekrar edilen örnekler ağın parametrelerini değiştirir ve bu değişim **benzer durumlara genelleşir** — DQN'in başarısının bir ölçüsü budur. Doğru karşıtlık şudur: tampon yalnız **yaşanmış** örnekleri üretir (dağılımı geçmişe bağlı); model **görülmemiş** $(s,a)$'lar için de öngörü üretir ve simüle örneklemin dağılımını plan seçebilir. Deterministik ortamda tablo model ile tamponun taşıdığı bilgi aynıdır; stokastik ortamda sayım modeli tek gözlemi değil **beklenen geçişi** taşır (pratik kartlarında bunu kanıtlıyorsunuz).
:::

**Değişen ortam: Dyna-Q+ ve öncelikli tarama.** Model öğrenilmişse ve ortam değişmişse, eski model hâlâ "çalışıyordur" ve ajan değişimi fark etmez: kısayol deneyinde (Sutton ve Barto Örnek 8.3) 3000. adımda sağdan bir geçit açılır, ama eski yol hâlâ hedefe götürdüğü için düz Dyna-Q kısayolu hiç aramaz — ölçülen ikinci yarı ödülü Dyna-Q için 165, keşif bonuslu Dyna-Q+ için 231 (oran 1.40; labda canlı). **Dyna-Q+** planlamada ödüle $\kappa\sqrt\tau$ bonusu ekler; $\tau$, o $(s,a)$ çiftinin son gerçek denenmesinden geçen adım sayısıdır. Uzun süredir denenmeyen geçiş, modelde "taze değil" sayılır ve planlama onu yeniden denemeye itilir. Bedeli ölçülür: değişmeyen ilk yarıda Dyna-Q+ 116 ödül topladı, Dyna-Q 136 — gereksiz keşif, ödül kaybıdır. İkinci ek yöntem **öncelikli tarama** (Moore & Atkeson 1993) planlama kuyruğunu rastgele değil, beklenen öğrenme büyüklüğüne ($|\delta|$) göre sıralar: x2 alıştırmasında aynı labirent için düz Dyna'nın bulduğu güncelleme sayısını belirgin biçimde düşürüyor.

### Şimdiki durumdan planlama: MPC {#mpc}

Dyna, rastgele görülmüş durumlarda planlar. **Şimdiki durumdan planlamak** daha hedefli: şu an bulunduğun $s$'ten ileri bak, bir sonraki eylemi iyileştir. Sürekli kontrolde bunun standart hâli **model öngörülü kontrolüdür** (MPC): her adımda $K$ eylem dizisi öner, her birini modelde $H$ adım ilerlet, maliyeti en küçük olan dizinin **ilk eylemini** uygula, bir adım at, tekrarla. Basit ve etkili bir çeşidi **rastgele atıştır** (random shooting): dizileri önsel olasılıkla değil rastgele örnekleyip en iyisini seçmek — şaşırtıcı biçimde iyi çalışır (kaos, ışınlanmış rastgele arama gibi). d aşaması tam bunu yapıyor: sinir ağı modeliyle CartPole'da 500 rastgele ikili dizi, $H=20$ ufuk; direği düşürmeyen en uzun dizinin ilk itmesi seçiliyor.

MPC'nin zayıf yanı çok adımlı model hatasıdır: $H$ adımlı ilerletmede tek adım hatası birikir. Ufuku kısaltmak hatayı sınırlar ama miyopluğu artırır; aradaki denge uygulamaya bağlıdır. Modelli hız + modellsiz sağlamlık birleştirilebilir: önce modelle öğren, sonra modellsiz yöntemle ince ayar (Nagabandi ve ark. 2018).

### Ağaç araması: üç fikirle işlenebilirlik {#agac-aramasi}

Ayrık karar probleminde "şimdiki durumdan plan" bir **arama ağacıdır**: kök şu anki durum, dallar eylemler, şans düğümleri olası $s'$'ler. Tam ağaç iki nedenden patlar: dallanma (Go'da ilk hamlede 361 dal) ve derinlik. Slayt 16'nın özü, ağacı üç yaklaşımla budamaktır:

- **Yaprak düğümleri:** değeri tam hesaplamak yerine varsayılan politika $\pi$ ile oyun sonuna kadar oynayıp getirinin ortalamasını al: $Q^*(s,a)\approx Q^\pi(s,a)\approx\frac1{n(s,a)}\sum_kG_k$. Yaklaşıklık kalitesi varsayılan politikanın kalitesine bağlıdır; AlphaGo'da varsayılan politika yerine bir **değer ağı** kullanılır (M11.11).
- **Şans düğümleri:** beklentiyi kapalı hesaplamak yerine geçiş modelinden **örnekleyerek** yaklaş: $Q^*(s,a)\approx R(s,a)+\gamma\frac1{n(s,a)}\sum_{s'\sim P(\cdot\mid s,a)}V(s')$.
- **Karar düğümleri:** bütün eylemleri genişletmek yerine yalnız umut vereni genişlet. Hangisi umutlu? Bu, her karar düğümünde bir **çok kollu haydut** problemidir (M11.7): kollar eylemler, her çekiliş bir simülasyon, ödül simülasyonun sonucu. UCB1'in cevabı oradaydı: $\bar Q(s,a)+c\sqrt{2\ln n(s)/n(s,a)}$.

:::uyarı Kaynak düzeltmesi (lect_12): düğüm değeri bonusla seçilen eylemin skoru değildir
Slayt 16, $a^*=\arg\max_a\bigl[Q(s,a)+c\sqrt{2\ln n(s)/n(s,a)}\bigr]$ tanımladıktan sonra $V^*(s)=Q(s,a^*)$ yazıyor. Bu tanım yanlış bağ kurar: $a^*$ **keşif bonusuyla** seçilmiş eylemdir ve bu seçim ölçütü yalnız *hangi çocuğu derinleştireceğimize* karar verir. Düğümün geri yayılan değeri, çocukların ortalama sonuçlarından gelir; oyuncu sırası düğümündeyse $V(s)=\max_a\bar Q(s,a)$ (ya da ziyaret ağırlıklı ortalama) tanımlanır. Bonusu değere katmak, keşif amaçlı cezayı politikanın gerçek değerine karıştırmak olur — bonus büyükken kötü bir eylem "değerli" görünür ve üst düğümlerin Q̄'ları şişer.
:::

Bu üç budama birlikte **Monte Carlo ağaç aramasını** (MCTS) verir. Seçim kuralı UCB1 tabanlı olduğundan algoritmaya **UCT** (Kocsis & Szepesvári 2006) denir.

:::türetim UCT neden haydut azaltımıdır
Karar düğümü $s$'te her eylem $a$, ortalaması bilinmeyen bir ödül dağılımına sahip bir koldur: ödül, $a$'dan başlayan simülasyonun sonucudur. M11.7'de Hoeffding eşitsizliğiyle $|\bar Q(s,a)-Q(s,a)|\le c\sqrt{2\ln N/n(s,a)}$ olasılıkla küçük demiştik; UCB1 bu genişliği kola ekleyerek "belirsiz ama umutlu" kollarla "bilinen iyi" kolları dengeler. MCTS'te fark, kolların **bağımlı** olmasıdır: bir kolu çekmek ağacı derinleştirir, diğer kolun ortalamasını dolaylı değiştirir. Yine de her düğümde yerel haydut muamelesi yapmak, UCT'nin tutarlılık sonucunu verir: simülasyon sayısı sonsuza giderken kökteki ziyaretler en iyi hamle(ler)e yoğunlaşır (iki kişili sıfır toplamlı, kesin bilgi oyunlarında; bağda kalan en iyi hamleler arasında bölünebilir). $c$ küçüldükçe keşif azalır, büyüdükçe ziyaretler dağılır — ölçülecek.
:::

### MCTS: dört adım {#mcts}

Her simülasyon dört adımdan oluşur ve değerleri X'in bakışından (kazanır +1, kaybeder −1, beraberlik 0) köke geri taşır:

1. **Seçim:** kökten, tamamen genişletilmiş düğümlerde UCT skoruyla in: $a=\arg\max_a\bigl[\bar Q(s,a)\cdot\text{oyuncu}+c\sqrt{\ln n(s)/n(s,a)}\bigr]$. Oyun sırası O'daysa işaret döner (O, X'in değerini küçültmek ister): skor $\bar Q\cdot p$, $p\in\{+1,-1\}$.
2. **Genişletme:** ilk kez varılan düğümde tek yeni çocuk ekler (bir hamle), simülasyonu oradan sürdürür.
3. **Simülasyon:** yeni çocuk terminal değilse varsayılan politika (burada düz rastgele oyun; e aşamasında `torch.multinomial` ile paralel) sonuç üretir; terminalse kesin değer kullanılır. Simülasyon başına birden çok rastgele oyun oynayıp ortalamak kestirimi yumuşatır.
4. **Geri yayılım:** sonucu yoldaki bütün düğümlere ekler: $N\leftarrow N+1$, $W\leftarrow W+\text{değer}$. Kökteki hamle seçimi **en çok ziyaret edilen** hamledir (skoru en yüksek olan değil — ziyaret sayısı, iyinin kanıtlanmışlığına daha sağlam bakar).

Ağaç kalitesi iki noktaya duyarlıdır ve ikisi de ölçülebilir: varsayılan politikanın kalitesi (yaprak değerlerinin gürültüsü) ve keşif katsayısı $c$.

<figure class="fig wide"><svg viewBox="0 0 640 340" role="img" aria-label="UCT ağacı: kök düğümünden beş hamleye dallar; daire alanı ziyaret sayısıyla orantılı, en çok ziyaret edilen kazandıran hamle kare 6."><text x="62.0" y="34.0" class="sm" text-anchor="middle">konum (X oynar)</text><line x1="26" y1="44" x2="98" y2="44" class="mute"/><line x1="26" y1="68" x2="98" y2="68" class="mute"/><line x1="26" y1="92" x2="98" y2="92" class="mute"/><line x1="26" y1="44" x2="26" y2="116" class="mute"/><line x1="50" y1="44" x2="50" y2="116" class="mute"/><line x1="74" y1="44" x2="74" y2="116" class="mute"/><rect x="26" y="44" width="72" height="72" class="mute"/><text x="38.0" y="61.0" class="tb" text-anchor="middle">O</text><text x="38.0" y="85.0" class="tb" text-anchor="middle">O</text><text x="62.0" y="85.0" class="ta" text-anchor="middle">X</text><text x="86.0" y="109.0" class="ta" text-anchor="middle">X</text><circle cx="190" cy="140" r="26" class="ca-f"/><text x="190.0" y="141.0" class="tb" text-anchor="middle">kök</text><text x="190.0" y="156.0" class="sm" text-anchor="middle">N = 200</text><line x1="216" y1="140" x2="332.0" y2="44" class="ca w2"/><circle cx="350" cy="44" r="18.0" class="ca-f"/><text x="350.0" y="48.0" text-anchor="middle">6</text><text x="380.0" y="42.0" class="sm" text-anchor="start">kare 6 · N = 175 · ort. değer 0.872</text><text x="380.0" y="56.0" class="sm" text-anchor="start">seçim skoru 1.12</text><line x1="216" y1="140" x2="342.2204830841108" y2="100" class="mute"/><circle cx="350" cy="100" r="7.8" class="mute-f"/><text x="350.0" y="104.0" text-anchor="middle">2</text><text x="369.8" y="98.0" class="sm" text-anchor="start">kare 2 · N = 8 · ort. değer −0.2</text><text x="369.8" y="112.0" class="sm" text-anchor="start">seçim skoru 0.939</text><line x1="216" y1="140" x2="342.4" y2="156" class="mute"/><circle cx="350" cy="156" r="7.6" class="mute-f"/><text x="350.0" y="160.0" text-anchor="middle">1</text><text x="369.6" y="154.0" class="sm" text-anchor="start">kare 1 · N = 7 · ort. değer −0.221</text><text x="369.6" y="168.0" class="sm" text-anchor="start">seçim skoru 0.997</text><line x1="216" y1="140" x2="342.5928677405914" y2="212" class="mute"/><circle cx="350" cy="212" r="7.4" class="mute-f"/><text x="350.0" y="216.0" text-anchor="middle">7</text><text x="369.4" y="210.0" class="sm" text-anchor="start">kare 7 · N = 6 · ort. değer −0.217</text><text x="369.4" y="224.0" class="sm" text-anchor="start">seçim skoru 1.1</text><line x1="216" y1="140" x2="343.03458474035205" y2="268" class="mute"/><circle cx="350" cy="268" r="7.0" class="mute-f"/><text x="350.0" y="272.0" text-anchor="middle">5</text><text x="369.0" y="266.0" class="sm" text-anchor="start">kare 5 · N = 4 · ort. değer −0.55</text><text x="369.0" y="280.0" class="sm" text-anchor="start">seçim skoru 1.06</text><text x="320.0" y="326.0" class="sm" text-anchor="middle">daire alanı ziyaret sayısıyla orantılı; kırmızı: minimax ile kazandıran tek hamle</text></svg>
<figcaption>Şekil 2 — X'in engellemesi gereken konumda (O··/OX·/··X) 200 simülasyonluk UCT koşusunun kök düğümü ve çocukları (c = 1.4, yaprak başına 20 rastgele oyun; hesaplandı). Kare 6 tek kazandıran hamle: 175 ziyaret ve ortalama değer 0.872 aldı; diğer hamlelerin hepsi O'nun sol sütunu tamamlamasına izin verir. Daire alanı ziyaret sayısıyla orantılı; seçim skoru ortalama değer + c·√(ln N/n) işlemidir.</figcaption></figure>


Şekil 2'deki konum güzel bir ölçüm dersi: O tehdit kurmuş (sol sütun), X oynayacak. İlk bakışta X'in yapması gereken "engellemek" gibi görünür ve analiz burada biter sanılır — oysa tam arama (minimax) daha fazlasını söylüyor: tek kazandıran hamle kare 6 yani engelleme hamlesinin kendisi; çünkü engellerken **çift tehdit** kurar (2-4-6 çaprazı ve 6-7-8 alt sıra) ve O yalnız birini kapatabilir. UCT, 200 simülasyonun 175'ini kendi kendine bu hamleye harcamış, ortalama değerini 0.87 bulmuş; diğer dört hamlenin değerleri −0.2 ile −0.55 arasında (hepsi O'nun kazanmasına izin veriyor). Rastgele oyunların getiri oranı X için yaklaşık 0.297'dir (el hesabındaki 105 oyun ölçümü) — ağaç olmasa yaprak değerleri bu civarda kalırdı ve engelleme hamlesini seçmek için yeterince ayrım oluşmazdı.

::lab l2

Deneydeki ölçümlerden ikisi özellikle öğretici:

- **$c$'nin iki ucu.** c = 0'da (bonus yok) bu konumda pay 0.98 çıktı — farklar büyük olduğundan ilk izlenim yeterli oldu. Ama yaprak başına tek rastgele oyunla (roll = 1) 12 tohumun 2'sinde pay 0.005'e düştü: kötü ilk izlenim bırakan hamle bir daha hiç seçilemez. c = 3'te ağaç genişledi ve UCT payı 12 tohumda 0.63–0.66 bandına indi (c = 1.4'te 0.85–0.89). Bonus ne çok ne az.
- **Önsel kalitesi.** PUCT varyantı (seçim bonusu $c\,\pi(a\mid s)\sqrt{n(s)}/(1+n(s,a))$) merkez/köşe ağırlıklı bir önselle çağrıldığında engelleme konumunda payı 0.87'den 0.95'e çıkardı — önsel doğruysa keşif hızlanır. Kazandıran hamlenin bir kenar olduğu ikinci konumda önsel yanlıştı ama ölçülen pay yine UCT'ten yüksek çıktı (0.92'ye karşı 0.875): hamle oyunu bitirdiğinden yaprak değeri kesindir ve kesin değerler yanlış önseli düzeltir. Küçük oyunlarda önselin cezası sınırlı; belirsizliğin ağaçta kaldığı büyük oyunlarda (Go) önsel kalitesi belirleyicidir.

Boş tahtada minimax değeri beraberliktir ve **bütün** dokuz ilk hamle bu değeri korur; ölçülen ziyaretler yine de merkezde (48/200) ve köşelerde yoğunlaşır çünkü rastgele oyunlarda bunlar daha çok kazandırır. "En çok ziyaret" ölçütü, değer eşitken bile amaca hizmet eden bir ayrım üretir.

### AlphaGo'nun arama ağacı {#alphago}

Slayt 18–19, AlphaGo'nun (Silver ve ark. 2016) iskeletini verir; ayrıntıları M11.11 işliyor. Dört aşama: (1) politika ağının insan oyunlarından **denetimli** öğrenimi, (2) politika ağının kendi kendine oyunlarla politika gradyanıyla (M11.9) iyileştirilmesi, (3) değer ağının politika ağının kendi kendine oyunlarından öğrenimi, (4) aramada her iki ağın kullanımı. Arama, MCTS'tir; fark iki yerdedir:

- **Simülasyon yerine değer ağı:** yaprak değeri, düz rastgele oyun yerine $v_\theta(s)\approx V^\pi(s)$ ile kestirilir (ya da rastgele oyunlarla karıştırılır — en iyi sonuç karışımınkidir).
- **Seçim kuralı politika önseliyle:** her kenarda $Q(s,a)$, $\pi(a\mid s)$ (politika ağının olasılığı) ve $n(s,a)$ saklanır ve $a^*=\arg\max_a\bigl[Q(s,a)+u(s,a)\bigr]$ seçilir.

:::uyarı Kaynak düzeltmesi (lect_12): PUCT bonusunda √N çarpanı eksik
Slayt 19, keşif bonusunu $u(s,a)\propto\pi(s,a)/(1+n(s,a))$ yazıyor. AlphaGo makalesinin denklemi (2) şudur: $u(s,a)\propto P(s,a)\,\dfrac{\sqrt{n(s)}}{1+n(s,a)}$, ayrıca makalede $c$ bir katsayıdır ve $n(s)=\sum_b n(s,b)$ kökün toplam ziyaretidir. $\sqrt{n(s)}$ çarpanı işlevsel değildir: onsuz bonus her zaman küçük kalır ve derinleşen ağaçta keşif ölülebilir — ziyaret arttıkça bonusun da büyümesi gerekir ki hiç denenmeyen kol zamanla çekilebilsin. El hesabındaki PUCT alanı (0.2 + 1·0.3·√100/(1+4) = 0.8) makaledeki formülle hesaplanmıştır; slayttaki eksik formülle 0.26 çıkardı ve 0.8 ölçüsüyle uyuşmazdı.
:::

Bu formülün haydut kökleri de nettir: UCT'nin $c\sqrt{\ln N/n}$ terimi önsel bilgisi olmayan kollar içindir; $P(s,a)\sqrt{N}/(1+n)$ terimi ise kolu **önsel olasılıkla ağırlandırır** — belirsizliğin neyi önceden iyice bildiğinden ayrılması. PUCT adıyla bu biçim AlphaGo Zero'da da (Silver ve ark. 2017) aramanın tek keşif mekanizmasıdır.

### Özet {#ozet}

- Modelli RL geçiş ve ödül modelini açıkça öğrenir ve onunla planlar: örnek verimliliği artar, karmaşıklık ve model hatası bedel olur. Döngü: eylem → gözlem → modeli güncelle → planla.
- Tablo model sayımlardan gelir: $\hat P=n(s,a,s')/n(s,a)$ (en çok olabilirlik), $\hat R$ gözlenen ödüllerin ortalaması. Kesinlik eşdeğerli planlama modeli gerçek bilip değer yinelemesini çözmektir; pahalıdır.
- Büyük durum uzayında model fonksiyon yaklaşımı olur (doğrusal, Gauss süreci, sinir ağı; artık tahmini). Çok adımlı ön bakışta model hatası birikir; MPC'de ufuk kısılır, modellsiz ince ayarla birleştirilir.
- Dyna-Q: her gerçek adımdan sonra $n$ simüle Q-öğrenme adımı; planlama çiftleri **görülmüş** $(s,a)$'lardan seçilir. Labirentte ikinci bölüm n = 0 için ortalama 752 adım, n = 50 için 98 (30 çalıştırma); dağılım ağır kuyrukludur (37–212).
- Tampon gerçek örnekleri tekrarlar, model görülmemiş $(s,a)$'lara da öngörü üretir; tablo Q'da bilgi aynı, sinir ağı Q'da ikisi de genelleşir.
- Değişen ortamda Dyna-Q+ denenmemişliğe $\kappa\sqrt\tau$ bonusu verir: kısayolu bulur (ikinci yarı 231'e karşı 165), değişmeyen ortamda keşif bedeli öder (116'ya karşı 136). Öncelikli tarama kuyruğu $|\delta|$ ile sıralar.
- Şimdiki durumdan planlama: MPC (rastgele atış) ve ayrıkta MCTS. MCTS üç budamayı birleştirir: yaprakta varsayılan politika, şans düğümünde örnekleme, karar düğümünde UCT ile umutlu kolu genişletme. Düğümün değeri çocuk ortalamalarındandır; keşif bonusu yalnız seçim içindir.
- MCTS'te kök seçimi en çok ziyaretli hamledir. Ölçülen duyarlılıklar: c = 0 gürültülü kestimde kilitlenmeye (12 tohumda 2 tam başarısızlık), c = 3 dağınıklığa (pay 0.63–0.66) götürür; iyi bir önsel payı 0.87'den 0.95'e çıkarır, kesin yaprak değerleri yanlış önseli düzeltir.
- AlphaGo: politika + değer ağıyla MCTS; yaprakta değer ağı, seçimde $u\propto P(s,a)\sqrt{n(s)}/(1+n(s,a))$ (slayttaki biçimde $\sqrt{n(s)}$ çarpanı eksikti).
