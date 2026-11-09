### Model, veri ve kısa bir tarih {#giris}

**Model** nedir? En basit örnek: gürültülü bir 5.0 voltluk ölçüm dizisi. "Sinyal sabit bir DC değer artı gürültüdür" demek bir modeldir; tek **parametresi** o sabit değerdir ve veriden kestirilir (MLE ile örnek ortalaması, M0.4). Gürültü güçlendikçe aynı model daha belirsiz bir kestirim verir. Bu derste modeller milyonlarca parametreli olacak, ama fikir aynı: veri bir olasılık dağılımından gelir; öğrenmek, o dağılımı (ya da ondan türeyen bir şeyi — sınıf olasılığı, regresyon değeri) parametreli bir modelle **yaklaşıklamaktır**.

- **Parametrik yaklaşım:** veri $x$'in yoğunluğunu $\theta$ ile modelle, $\theta$'yı bul. **En çok olabilirlik (MLE):** $\arg\max_\theta p(x\mid\theta)$ — parametreler bilinmeyen ama deterministik sayılar; yalnız veriye dayanır, veri sonsuza giderken tutarlıdır. **Bayesçi yaklaşım:** $\theta$ da rastgeledir; önsel bilgi $p(\theta)$ eklenir. Tek bir nokta kestirimi istenirse $\arg\max_\theta p(x\mid\theta)p(\theta)$ (MAP, M0.4); tam Bayesçi çıkarım ise tüm sonsal dağılımı tutar ve çoğu zaman hesaplanamaz integraller gerektirir.
- **Üretici modeller** (dil modelleri, difüzyon) veri dağılımını o kadar iyi modeller ki ondan **örnekleyip** yeni veri üretebilirsin (Faz 5).

:::örnek Görüntü uzayı ne kadar büyük?
200 × 200 RGB, kanal başına 8 bit: 40 000 piksel, her piksel $256^3=2^{24}$ değer alabilir. Olası görüntü sayısı $\big(2^{24}\big)^{40\,000}=2^{960\,000}\approx10^{289\,000}$ — evrendeki atom sayısının ($\sim10^{80}$) çok ötesinde. Kediler, köpekler ve tüm "doğal" görüntüler bu uzayın akıl almaz derecede küçük bir bölgesinde kümelenir; bir modeli öğrenmek, o bölgeyi (veri dağılımını) yaklaşıklamaktır.
:::

**Derin öğrenme** = geniş ve derin sinir ağlarını büyük veriye uygulama pratiği; dört aşaması: (1) mimari tasarımı, (2) **eğitim** (parametreleri veriden öğrenmek; çoğu zaman çevrimdışı, süre ikincil), (3) **çıkarım** (öğrenilmiş modeli yeni veriye uygulamak; çoğu zaman çevrimiçi, gecikme ve hesap kritik), (4) dağıtım ve bakım (MLOps, sürekli öğrenme).

**Kısa tarih.** 1943 McCulloch–Pitts nöron modeli → 1957 Rosenblatt perceptronu → 1969 Minsky–Papert (XOR) ve **ilk kış**: uygulanabilir bir eğitim yöntemi ve hesap gücü yok → 1970 Linnainmaa ters mod otomatik türev → 1986 Rumelhart, Hinton ve Williams geri yayılımı yaygınlaştırır → 1990'lar **ikinci kış**: modeller yetersiz, donanım yetmiyor, SVM'ler baskın → 2000'ler sonu programlanabilir GPU'lar (CUDA) → 2012 AlexNet (60 milyon parametre, 650 000 nöron) → 2014 DeepFace → 2016 AlphaGo → 2020 GPT-3 → 2021 AlphaFold → 2022 Stable Diffusion → 2023 Segment Anything → 2024 V-JEPA.

Sinir ağları beyinden **esinlenmiştir** — ama modern modellerin çoğu biyolojik olarak inandırıcı değildir. Esin, aşağıdaki en basit modellerde en belirgindir.

### Biyolojik nöron ve McCulloch–Pitts modeli {#mp}

Biyolojik nöron, beyin ile vücudun geri kalanı arasında bilgi taşıyan bir sinir hücresidir. Kısa elektrik darbeleri (**aksiyon potansiyelleri**) üretir; bunlar akson boyunca ilerler ve uçtaki **sinapslarda** kimyasal sinyallerin (nörotransmitterler) salınmasına yol açar. Bir nöron birkaç milisaniye içinde yeterince nörotransmitter alırsa kendi darbesini üretir (ateşler). Ama bazı nörotransmitterler ateşlemeyi **engeller** (inhibitör). Tek tek nöronlar basit davranır; gücü örgütlenmeden gelir: milyarlarcası, her biri binlerce başkasına bağlı bir ağ oluşturur, ve yeterince basit birimlerden oluşan bir ağın çok karmaşık hesaplar yapabildiği matematiksel olarak gösterilebilir. Beyin kabuğunun büyük kısmını (insanda %90'ı) oluşturan **neokorteks** — algı, biliş, motor komutlar, uzamsal akıl yürütme ve dil — ardışık **katmanlar** hâlinde örgütlenmiştir; yapay ağların katmanlı yapısının esin kaynağı budur.

:::tanım McCulloch–Pitts nöronu (1943)
Bir ya da daha çok **ikili** girdi (açık/kapalı), tek ikili çıktı. Girdilerinden belli bir sayıdan fazlası etkin olduğunda çıktısını etkinleştirir. İnhibitör bir bağlantı etkinse nöron ateşlemez.
:::

Bu kadar basit bir modelle bile **her** mantık önermesini (VE, VEYA, DEĞİL…) hesaplayan bir ağ kurulabilir. Nöronun en az iki girdisi etkinken ateşlediğini varsay:

<figure class="fig wide"><svg viewBox="0 0 600 215" role="img" aria-label="McCulloch–Pitts nöronlarıyla dört mantık işlemi: özdeşlik, VE, VEYA ve A VE DEĞİL B"><defs><marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><text x="60" y="22" text-anchor="middle" class="tp">C = A</text><circle cx="60" cy="160" r="14" class="cp w2"/><text x="60" y="165" text-anchor="middle">C</text><text x="210" y="22" text-anchor="middle" class="tp">C = A ∧ B</text><circle cx="210" cy="160" r="14" class="cp w2"/><text x="210" y="165" text-anchor="middle">C</text><text x="360" y="22" text-anchor="middle" class="tp">C = A ∨ B</text><circle cx="360" cy="160" r="14" class="cp w2"/><text x="360" y="165" text-anchor="middle">C</text><text x="510" y="22" text-anchor="middle" class="tp">C = A ∧ ¬B</text><circle cx="510" cy="160" r="14" class="cp w2"/><text x="510" y="165" text-anchor="middle">C</text><circle cx="60" cy="70" r="14" class="ca w2"/><text x="60" y="75" text-anchor="middle">A</text><line x1="66.0" y1="84.0" x2="66.0" y2="144.0" class="ink" marker-end="url(#a1)"/><line x1="54.0" y1="84.0" x2="54.0" y2="144.0" class="ink" marker-end="url(#a1)"/><circle cx="170" cy="70" r="14" class="ca w2"/><text x="170" y="75" text-anchor="middle">A</text><circle cx="250" cy="70" r="14" class="ca w2"/><text x="250" y="75" text-anchor="middle">B</text><line x1="175.7" y1="82.8" x2="203.5" y2="145.4" class="ink" marker-end="url(#a1)"/><line x1="244.3" y1="82.8" x2="216.5" y2="145.4" class="ink" marker-end="url(#a1)"/><circle cx="320" cy="70" r="14" class="ca w2"/><text x="320" y="75" text-anchor="middle">A</text><circle cx="400" cy="70" r="14" class="ca w2"/><text x="400" y="75" text-anchor="middle">B</text><line x1="330.3" y1="80.8" x2="358.1" y2="143.3" class="ink" marker-end="url(#a1)"/><line x1="398.9" y1="84.8" x2="371.1" y2="147.4" class="ink" marker-end="url(#a1)"/><line x1="321.1" y1="84.8" x2="348.9" y2="147.4" class="ink" marker-end="url(#a1)"/><line x1="389.7" y1="80.8" x2="361.9" y2="143.3" class="ink" marker-end="url(#a1)"/><circle cx="470" cy="70" r="14" class="ca w2"/><text x="470" y="75" text-anchor="middle">A</text><circle cx="550" cy="70" r="14" class="ca w2"/><text x="550" y="75" text-anchor="middle">B</text><line x1="480.3" y1="80.8" x2="508.1" y2="143.3" class="ink" marker-end="url(#a1)"/><line x1="471.1" y1="84.8" x2="498.9" y2="147.4" class="ink" marker-end="url(#a1)"/><line x1="544.3" y1="82.8" x2="517.3" y2="143.6" class="cq w2"/><circle cx="517.3" cy="143.6" r="4" class="cq-f"/><text x="300" y="205" text-anchor="middle" class="sm">her C nöronu: en az 2 etkin (uyarıcı) bağlantı → 1; inhibitör (●) bağlantı etkinse → 0</text></svg><figcaption>Şekil 1 — McCulloch–Pitts nöronlarıyla mantık. Her nöron, en az iki girdi bağlantısı etkin olduğunda ateşler. Özdeşlik: A'dan iki bağlantı. VE: A ve B'den birer. VEYA: her birinden iki (biri yeterli). A ∧ ¬B: A'dan iki uyarıcı, B'den bir <b>inhibitör</b> bağlantı — B etkinse nöron ateşlemez (Géron, Bölüm 10'daki örneklerin yeniden çizimi).</figcaption></figure>

### Eşik mantık birimi (TLU) ve perceptron {#tlu}

Rosenblatt'ın 1957'de icat ettiği **perceptron**, en basit yapay sinir ağı mimarilerinden biridir. Biraz farklı bir yapay nörona dayanır: **eşik mantık birimi** (threshold logic unit, TLU; ya da doğrusal eşik birimi, LTU). Girdiler ve çıktı ikili değil **sayıdır** ve her girdi bağlantısının bir **ağırlığı** vardır.

<figure class="fig"><svg viewBox="0 0 480 232" role="img" aria-label="Eşik mantık birimi: girdiler ağırlıklarla çarpılıp toplanır, sonuca basamak fonksiyonu uygulanır"><defs><marker id="t1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="t2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cp-f"/></marker></defs><circle cx="40" cy="35" r="14" class="ca w2"/><text x="40" y="40" text-anchor="middle">x₁</text><line x1="52.8" y1="40.6" x2="185.3" y2="98.6" class="ink" marker-end="url(#t1)"/><text x="110" y="64" class="sm">w₁</text><circle cx="40" cy="85" r="14" class="ca w2"/><text x="40" y="90" text-anchor="middle">x₂</text><line x1="53.9" y1="86.7" x2="184.1" y2="103.0" class="ink" marker-end="url(#t1)"/><text x="110" y="89" class="sm">w₂</text><circle cx="40" cy="135" r="14" class="ca w2"/><text x="40" y="140" text-anchor="middle">x₃</text><line x1="53.8" y1="132.4" x2="184.3" y2="107.9" class="ink" marker-end="url(#t1)"/><text x="110" y="114" class="sm">w₃</text><circle cx="40" cy="185" r="14" class="mute w2"/><text x="40" y="190" text-anchor="middle">1</text><line x1="52.5" y1="178.7" x2="185.7" y2="112.2" class="ink" marker-end="url(#t1)"/><text x="110" y="160" class="sm">b</text><circle cx="215" cy="105" r="24" class="ink w2"/><text x="215" y="102" text-anchor="middle">Σ</text><text x="215" y="118" text-anchor="middle" class="sm">z</text><rect x="270" y="81" width="64" height="48" rx="6" class="cp w2"/><path d="M282 118 H302 V92 H322" class="cp w2" fill="none"/><line x1="239.0" y1="105.0" x2="268.0" y2="105.0" class="ink" marker-end="url(#t1)"/><line x1="334.0" y1="105.0" x2="378.0" y2="105.0" class="cp" marker-end="url(#t2)"/><text x="388" y="110">ŷ = step(z)</text><text x="215" y="160" class="sm">z = w₁x₁ + w₂x₂ + w₃x₃ + b = wᵀx + b</text><text x="20" y="222" class="sm">girdi nöronları (aynen geçirir) ve bias nöronu (hep 1)</text></svg><figcaption>Şekil 2 — Eşik mantık birimi (TLU). Girdiler ve çıktı ikili değil sayıdır; her bağlantının bir ağırlığı vardır. TLU ağırlıklı toplamı hesaplar ve basamak fonksiyonu uygular: heaviside (z ≥ 0 → 1, değilse 0) ya da işaret (z ≥ 0 → +1, değilse −1).</figcaption></figure>

$$z=w^\top x+b,\qquad \hat y=\mathrm{step}(z),\qquad \mathrm{heaviside}(z)=\begin{cases}0,&z<0\\1,&z\ge0\end{cases},\quad \mathrm{sgn}(z)=\begin{cases}-1,&z<0\\+1,&z\ge0.\end{cases}$$

Tek bir TLU, ağırlıkları veriyle eğitilerek basit bir **doğrusal ikili sınıflandırıcı** olarak kullanılabilir: $w^\top x+b=0$ hiper düzleminin iki yanı iki sınıftır (lojistik regresyonun sınırıyla aynı geometri, M1.4; ama olasılık değil sert karar verir).

**TLU, McCulloch–Pitts nöronundan daha geneldir:** MP nöronunun yapabildiği her şeyi yapar. Örneğin VEYA: $w=(1,1)$, $b=-0.5$ ⇒ $(0,0)\to z=-0.5\to0$; diğer üç girdi $z\ge0.5\to1$. Bu çözüm **tek değildir**: $(1,1)$ ağırlıklarıyla $-1<b\le0$ aralığındaki her $b$ ve bunların pozitif katları da çalışır — öğrenme algoritmasının hangisini bulacağı başlangıca ve örneklerin sırasına bağlıdır.

**Perceptron**, tek katmanda örgütlenmiş bir ya da daha çok TLU'dan oluşur; her TLU **her** girdiye bağlıdır. Bir katmandaki tüm nöronlar önceki katmanın tüm nöronlarına bağlıysa katmana **tam bağlı** (fully connected) ya da **yoğun** (dense) katman denir. Girdiler, kendilerine verileni aynen çıkaran özel **girdi nöronlarına** beslenir (hepsi birlikte **girdi katmanı**); ek bir **bias özniteliği** de hep 1 çıkaran bir **bias nöronuyla** eklenir. İki girdili, üç çıktılı bir perceptron bir örneği aynı anda üç ayrı ikili sınıfa ayırabilir: **çok çıktılı** sınıflandırıcı.

:::tanım Bir katmanın matris biçimi
Birçok örnek için bir katmanın çıktıları tek bir matris işlemiyle hesaplanır (M0.2):
$$h_{W,b}(X)=\varphi(XW+b).$$
- $X$: girdi özniteliklerinin matrisi — **her örnek bir satır**, her öznitelik bir sütun ($m\times n$).
- $W$: bias nöronundan gelenler hariç tüm bağlantı ağırlıkları — **her girdi nöronu bir satır, katmandaki her yapay nöron bir sütun** ($n\times k$).
- $b$: bias nöronundan her yapay nörona giden ağırlıklar, nöron başına bir terim ($k$); toplama yayınla (broadcasting) her satıra eklenir.
- $\varphi$: **aktivasyon fonksiyonu**; TLU'larda basamak fonksiyonu.

Tek bir nöronun çıktısı da bir iç çarpımın (M0.1) aktivasyonudur: $\varphi(w\cdot x+b)$. PyTorch'un `nn.Linear` katmanı $W$'yi $(k,n)$ shape'inde saklar ve $xW^\top+b$ hesaplar — aynı işlem, devrik saklama.
:::

### Perceptron öğrenme kuralı {#kural}

Rosenblatt'ın eğitim algoritması büyük ölçüde **Hebb kuralından** esinlenmiştir. Donald Hebb (1949), bir biyolojik nöron başka bir nöronu sık sık tetiklediğinde aralarındaki bağlantının güçlendiğini öne sürdü: *"birlikte ateşleyen hücreler birbirine bağlanır"* (cells that fire together, wire together). İki nöron aynı anda ateşlediğinde aralarındaki bağlantı ağırlığı artma eğilimindedir. Perceptron bu kuralın, ağın tahmin **hatasını** hesaba katan bir varyantıyla eğitilir: hatayı azaltmaya yardım eden bağlantıları güçlendirir.

:::tanım Perceptron öğrenme kuralı
Perceptron'a eğitim örnekleri **birer birer** verilir; her örnek için tahmin yapılır. Yanlış tahmin yapan her çıktı nöronu için, doğru tahmine katkıda bulunacak girdilerden gelen bağlantılar güçlendirilir:
$$w_{i,j}^{(t+1)}=w_{i,j}^{(t)}+\eta\,(y_j-\hat y_j)\,x_i .$$
$w_{i,j}$: $i$. girdi nöronundan $j$. çıktı nöronuna ağırlık; $x_i$: o anki örneğin $i$. girdisi; $\hat y_j$: $j$. çıktı nöronunun tahmini, $y_j$: hedefi (0/1 ya da −1/+1); $\eta$: öğrenme oranı. Bias, hep 1 olan girdinin ağırlığı olarak aynı kuralla güncellenir.
:::

**Kural ne yapıyor?** Tahmin doğruysa $y_j-\hat y_j=0$: hiçbir şey değişmez. Hedef 1 ama tahmin 0 ise $w\leftarrow w+\eta x$: ağırlık vektörü $x$'e doğru döner, bir sonraki seferde $w^\top x$ daha büyük olur. Tersi ise $x$'ten uzaklaşır. ±1 etiketlerle (ve $\eta$ içine 2 çarpanını katarak) tek satır: **hata yapınca $w\leftarrow w+\eta\,y\,x$.**

:::türetim Perceptron kuralı = bir kaybın SGD'si
±1 etiketler, bias $x$'e gömülü. **Perceptron kriteri** $\ell(w;x,y)=\max(0,-y\,w^\top x)$: doğru sınıflanan örnek 0 ceza alır, yanlış sınıflanan örnek sınırdan uzaklığıyla orantılı ceza alır. Gradyanı: $y\,w^\top x>0$ iken 0; $y\,w^\top x<0$ iken $-y\,x$. SGD adımı $w\leftarrow w-\eta\nabla\ell=w+\eta\,y\,x$ — **yalnız hatalı örneklerde**. Yani perceptron, bu parçalı doğrusal kayıpta tek örnekli SGD'dir. Hinge kaybı $\max(0,1-y\,w^\top x)$ (SVM, M2.3) ise doğru ama sınıra **yakın** örnekleri de cezalandırır; fark marj kavramıdır.
:::

**Öğrenme oranı neden önemsiz?** $w_0=0$ ise her güncelleme $\eta$ çarpanı taşır, dolayısıyla $w_t=\eta\cdot(\text{η = 1 ile elde edilen } w_t)$: tüm ağırlıklar aynı pozitif sayıyla ölçeklenir, $\mathrm{sgn}(w^\top x)$ ve dolayısıyla **tüm tahminler aynıdır**. (Rastgele başlangıçta bu artık doğru değildir.)

### Yakınsama teoremi {#novikoff}

Perceptron'un en önemli kuramsal sonucu: veri doğrusal ayrılabilirse algoritma **sonlu sayıda** hatadan sonra durur. Sınır, ne örnek sayısına ne boyuta bağlıdır — yalnız verinin **geometrisine**.

:::türetim Novikoff teoremi (1962)
**Varsayımlar:** ±1 etiketler; bias $x$'e gömülü; $\|x_i\|\le R$ her $i$ için; veri $\gamma>0$ marjıyla ayrılabilir: $\|u\|=1$ olan bir $u$ için $y_i\,u^\top x_i\ge\gamma$ her $i$ için. Algoritma $w_0=0$'dan başlar ve yalnız hatada ($y\,w^\top x\le0$) $w\leftarrow w+y\,x$ günceller.

$k$ güncellemeden sonraki ağırlık $w_k$ olsun.
1. **$u$ yönünde ilerleme:** her güncellemede $u^\top w_k=u^\top w_{k-1}+y\,u^\top x\ge u^\top w_{k-1}+\gamma$. Tümevarımla $u^\top w_k\ge k\gamma$.
2. **Normun yavaş büyümesi:** $\|w_k\|^2=\|w_{k-1}\|^2+2y\,w_{k-1}^\top x+\|x\|^2\le\|w_{k-1}\|^2+R^2$, çünkü güncelleme yalnız hatada yapılır ($y\,w_{k-1}^\top x\le0$). Tümevarımla $\|w_k\|^2\le kR^2$.
3. **Birleştir** (Cauchy–Schwarz, $\|u\|=1$): $k\gamma\le u^\top w_k\le\|w_k\|\le\sqrt k\,R$. Buradan
$$\boxed{k\le\Big(\frac R\gamma\Big)^2.}$$

Sezgi: güncellemeler $w$'yi $u$ yönünde **doğrusal** ($\propto k$), boyunu ise ancak **karekök** ($\propto\sqrt k$) hızında büyütür; $w$ ile $u$ arasındaki açının kosinüsü $\ge\sqrt k\,\gamma/R$ olur ve 1'i geçemeyeceğine göre $k$ sınırlıdır.
:::

**Teorem neyi söylemez?** (1) Ayrılabilir olmayan veride hiçbir garanti yoktur: perceptron **sonsuza kadar dolaşır** ve son ağırlıklar rastgele bir anın ağırlıklarıdır (pratik çözümler: en iyi ağırlığı saklayan **cep algoritması**, ya da tüm adımların ağırlıklarının ortalamasını alan **ortalama perceptron**). (2) Bulunan ayırıcı herhangi bir ayırıcıdır, en büyük marjlı olan değil (o SVM'dir, M2.3). (3) Sınır kötü durum içindir; gerçek hata sayısı çoğu zaman çok daha küçüktür.

Aşağıda perceptronu farklı verilerde eğit. Ayrılabilir veride toplam hata sayısını Novikoff sınırıyla karşılaştır; boşluğu daralt ($\gamma$ küçülür); sonra XOR ve halka verisini dene.

::lab l1

### XOR ve doğrusal ayrılamazlık {#xor}

Tipik bir perceptron tek bir TLU'dur: birçok girdi, tek çıktı. **Karar sınırı doğrusaldır**; bu yüzden perceptron karmaşık örüntüleri öğrenemez. En ünlü örnek XOR (özel veya): $(0,0)\to0$, $(1,1)\to0$, $(0,1)\to1$, $(1,0)\to1$.

:::kanıt Hiçbir TLU XOR'u hesaplayamaz
$w_1x_1+w_2x_2+b\ge0\iff\hat y=1$ olsun. Dört koşul: $(0,0)$: $b<0$; $(1,1)$: $w_1+w_2+b<0$; $(1,0)$: $w_1+b\ge0$; $(0,1)$: $w_2+b\ge0$. Son ikisini topla: $w_1+w_2+2b\ge0$, yani $w_1+w_2+b\ge-b>0$ (çünkü $b<0$). Bu, ikinci koşulla çelişir. ∎ Aynı sonuç **her** doğrusal sınıflandırıcı için geçerlidir (lojistik regresyon dahil).
:::

Minsky ve Papert'in *Perceptrons* (1969) kitabı bu tür sınırlamaları ayrıntılı gösterdi. Araştırmacılar perceptronlardan çok daha fazlasını beklemişti; bazıları o kadar hayal kırıklığına uğradı ki sinir ağı araştırmasını tamamen bıraktı — birinci kış. İki çıkış yolu vardır:

**1. Öznitelikleri dönüştür.** XOR verisine üçüncü bir öznitelik $x_1x_2$ ekle — ya da ±1 kodlamada etiket tam olarak $\mathrm{sgn}(x_1x_2)$'dir: 3 boyutlu uzayda $w=(0,0,1)$ düzlemi veriyi ayırır. Merkezde bir sınıf, çevresinde öbürü olan bir "halka" verisi için $x_1^2+x_2^2$ eklemek yeterlidir: yarıçap eşiği bir düzlem olur. Orijinal uzayda karar sınırı artık doğru değil, hiperbol ya da çemberdir. Genel ilke (**Cover teoremi**, 1965): veriyi doğrusal olmayan bir dönüşümle yüksek boyutlu bir uzaya taşımak, rastgele bir etiketlemenin doğrusal ayrılabilir olma olasılığını artırır; $d$ boyutta (bias dahil) genel konumdaki $N\le d$ nokta **her** etiketlemede ayrılabilir. Çekirdek yöntemleri (SVM, M2.3) bu fikrin sistematik hâlidir. Sorun: doğru dönüşümü elle bulmak gerekir. Deneyde XOR verisini $x_1x_2$ ile, halka verisini $x_1^2+x_2^2$ ile dene — ve tersini.

**2. Katman ekle.** Perceptronların bazı sınırlamaları birden çok katman üst üste koyarak ortadan kalkar: **çok katmanlı perceptron** (MLP). Aşağıdaki MLP XOR'u çözer:

<figure class="fig"><svg viewBox="0 0 540 210" role="img" aria-label="XOR çözen iki katmanlı ağ: bir gizli birim VEYA, diğeri VE hesaplar; çıktı VEYA ve değil VE"><defs><marker id="x1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><circle cx="50" cy="60" r="14" class="ca w2"/><text x="50" y="65" text-anchor="middle">x₁</text><circle cx="50" cy="160" r="14" class="ca w2"/><text x="50" y="165" text-anchor="middle">x₂</text><circle cx="210" cy="60" r="14" class="ink w2"/><text x="210" y="65" text-anchor="middle">h₁</text><circle cx="210" cy="160" r="14" class="ink w2"/><text x="210" y="165" text-anchor="middle">h₂</text><circle cx="370" cy="110" r="14" class="cp w2"/><text x="370" y="115" text-anchor="middle">ŷ</text><line x1="64.0" y1="60.0" x2="194.0" y2="60.0" class="ink" marker-end="url(#x1)"/><line x1="61.9" y1="67.4" x2="196.4" y2="151.5" class="ink" marker-end="url(#x1)"/><line x1="61.9" y1="152.6" x2="196.4" y2="68.5" class="ink" marker-end="url(#x1)"/><line x1="64.0" y1="160.0" x2="194.0" y2="160.0" class="ink" marker-end="url(#x1)"/><text x="125" y="50" class="sm">1</text><text x="125" y="186" class="sm">1</text><text x="96" y="92" class="sm">1</text><text x="96" y="140" class="sm">1</text><line x1="223.4" y1="64.2" x2="354.7" y2="105.2" class="ink" marker-end="url(#x1)"/><line x1="223.4" y1="155.8" x2="354.7" y2="114.8" class="ink" marker-end="url(#x1)"/><text x="290" y="72" class="sm tq">+1</text><text x="290" y="160" class="sm">−1</text><text x="210" y="30" text-anchor="middle" class="sm">b = −0.5 (VEYA)</text><text x="210" y="198" text-anchor="middle" class="sm">b = −1.5 (VE)</text><text x="370" y="148" text-anchor="middle" class="sm">b = −0.5</text><text x="405" y="80" class="sm">ŷ = VEYA ∧ ¬VE</text><text x="405" y="96" class="sm">= XOR</text></svg><figcaption>Şekil 3 — XOR'u çözen çok katmanlı perceptron (heaviside TLU'lar). Gizli birimler h₁ = step(x₁ + x₂ − 0.5) (VEYA) ve h₂ = step(x₁ + x₂ − 1.5) (VE); çıktı ŷ = step(h₁ − h₂ − 0.5). (0,0) ve (1,1) için 0, (0,1) ve (1,0) için 1 verir.</figcaption></figure>

İki yol aslında aynı fikirdir: gizli katman, doğrusal sınıflandırıcının (çıktı TLU'su) ihtiyaç duyduğu **dönüşümü kendisi hesaplar** — ($h_1,h_2$) uzayında XOR doğrusal ayrılabilirdir: (0,0) → (0,0), (1,1) → (1,1), karışıklar → (1,0). MLP'nin son katmanı her zaman bir doğrusal sınıflandırıcıdır; önceki katmanlar o sınıflandırıcının işini kolaylaştıran gösterimi **öğrenir**. Derin öğrenmenin özü budur.

### Çok katmanlı perceptron ve derin ağlar {#mlp}

Bir MLP şunlardan oluşur: bir (aynen geçiren) **girdi katmanı**, **gizli katman** denen bir ya da daha çok TLU katmanı ve son bir TLU katmanı olan **çıktı katmanı**. Girdiye yakın katmanlara **alt**, çıktıya yakın olanlara **üst** katmanlar denir. MLP **ileri beslemelidir** (feedforward): veri tek yönde, girdiden çıktıya akar. Çıktı katmanı hariç her katman bir sonrakine tam bağlıdır (her katmanda bir bias nöronu). Gizli katmanlardan oluşan **derin bir yığın** içeren yapay sinir ağına **derin sinir ağı** (DNN) denir; derin öğrenme alanı DNN'leri ve daha genel olarak büyük veriyle eğitilen derin hesap yığınlarını inceler.

**Aktivasyon neden gerekli?** Doğrusal katmanları art arda bağlamak yine doğrusal bir dönüşüm verir:
$$h^{(2)}=W_2\big(W_1x+b_1\big)+b_2=\underbrace{W_2W_1}_{W}\,x+\underbrace{W_2b_1+b_2}_{b}.$$
Katmanlar arasında doğrusal olmayan bir işlem yoksa en derin yığın bile **tek bir katmana** denktir ve çok karmaşık problemleri çözemez. Tersine, doğrusal olmayan aktivasyonlu yeterince büyük bir ağ kuramsal olarak her sürekli fonksiyonu yaklaşıklayabilir (aşağıda). Doğrusal olmayan işlemler yalnız aktivasyonlar değildir: max pooling da öyledir (M4.1).

**Basamak fonksiyonundan sigmoid'e.** Rosenblatt'ın kuralı yalnız tek katmanlı ağları eğitebiliyordu; gizli katmanlar için gizli birimlerin "doğru cevabı" bilinmez. Gradyan tabanlı eğitim (geri yayılım, M3.2) için aktivasyonun türevi gerekir: basamak fonksiyonu yalnız düz parçalardan oluşur — türevi her yerde 0 (ve sıfırda tanımsız), gradyan inişi düz bir yüzeyde hareket edemez. Rumelhart ve arkadaşları basamağı **lojistik (sigmoid)** fonksiyonla değiştirdi: her yerde iyi tanımlı, sıfırdan farklı bir türevi vardır, gradyan inişi her adımda biraz ilerleyebilir. Geri yayılım başka aktivasyonlarla da çalışır; bugünün popüler seçimleri tanh ve ReLU'dur (M3.5).

:::uyarı Kaynak düzeltmesi: ileri ve geri geçiş
Ders slaytlarında "geri geçiş, maliyeti düşürecek biçimde ağırlık ve bias güncelleme adımıdır" deniyor. Daha kesin söylersek: **geri geçiş yalnız gradyanları hesaplar** (zincir kuralıyla, çıktıdan girdiye, M3.2–M3.3); **güncelleme** ondan sonra gelen ayrı bir optimizer adımıdır (`loss.backward()` ile `optimizer.step()` ayrımı). Geri yayılım tek başına bir eğitim algoritması değil, gradyan hesaplama yöntemidir; "geri yayılımla eğitim" = geri yayılımla hesaplanan gradyanlarla (S)GD.
:::

**Hatırlatma — eğitim döngüsü (M3.2):** parametreler rastgele başlatılır; her mini-batch (örneğin 32 rastgele örnek; iadeli ya da iadesiz seçilebilir) ileri geçişte ağdan geçirilir (tüm ara sonuçlar geri geçiş için saklanır), kayıp ölçülür, geri geçişte zincir kuralıyla her parametrenin hataya katkısı hesaplanır ve bir gradyan adımı atılır. Tüm eğitim setinden bir geçişe **epoch** denir; algoritma kayıp bir eşiğin altına inene ya da en fazla epoch sayısına ulaşılana kadar çalışır.

### Evrensel yaklaşım {#evrensel}

:::tanım Evrensel yaklaşım teoremi
$\varphi$ sürekli ve **polinom olmayan** bir aktivasyon olsun (sigmoid, tanh, ReLU…). $K\subset\mathbb R^d$ kompakt (kapalı ve sınırlı) bir küme, $f:K\to\mathbb R$ sürekli ve $\varepsilon>0$ ise **tek gizli katmanlı** ve sonlu $N$ nöronlu bir ağ
$$\hat f(x)=\sum_{k=1}^N c_k\,\varphi(a_k^\top x+b_k)$$
vardır ki $\sup_{x\in K}|f(x)-\hat f(x)|<\varepsilon$. (Cybenko 1989 sigmoidal aktivasyonlar için; Hornik 1991; Leshno ve ark. 1993 "polinom değil" koşuluyla.)
:::

Teorem güçlü ama dar: **varlık** söyler. $N$'nin ne kadar büyük olacağını (boyutla üstel olabilir), bu ağırlıkların gradyan inişiyle bulunup bulunamayacağını ve ağın eğitim verisi dışında iyi genelleyip genellemeyeceğini söylemez. Polinom aktivasyonların dışlanması da anlamlıdır: derecesi $p$ olan polinom aktivasyonlu tek katmanlı ağ en fazla $p$ dereceli bir polinomdur.

ReLU için tek boyutta teorem **yapıcı** olarak kanıtlanabilir — ve yaklaşım hatası da hesaplanabilir:

:::türetim ReLU menteşeleriyle parçalı doğrusal ara değerleme
$[0,1]$'i $K$ eşit aralığa böl: düğümler $t_k=k/K$, $h=1/K$. $f$'yi düğümlerde ara değerleyen parçalı doğrusal $\hat f$'nin $k$. parçadaki eğimi $s_k=\big(f(t_{k+1})-f(t_k)\big)/h$. $\mathrm{ReLU}(x-t_k)$, $t_k$'den önce 0, sonra eğimi 1 olan bir **menteşedir**; eğimi $t_k$'de $c_k=s_k-s_{k-1}$ kadar değiştirmek için ($s_{-1}=0$):
$$\hat f(x)=f(0)+\sum_{k=0}^{K-1}c_k\,\mathrm{ReLU}(x-t_k).$$
Bu, $K$ gizli ReLU'lu (ağırlık 1, bias $-t_k$) ve çıktı biası $f(0)$ olan tek gizli katmanlı bir ağdır; her düğümde $\hat f(t_k)=f(t_k)$.

**Hata:** $f$ iki kez sürekli türevliyse doğrusal ara değerlemenin bilinen hatası, $[t_k,t_{k+1}]$ aralığında $f(x)-\hat f(x)=\tfrac12f''(\xi)(x-t_k)(x-t_{k+1})$; $|(x-t_k)(x-t_{k+1})|\le h^2/4$ olduğundan
$$\max_x|f(x)-\hat f(x)|\le\frac{h^2}{8}\max|f''|=\frac{\max|f''|}{8K^2}.$$
Yani nöron sayısını ikiye katlamak hatayı dörde böler. $f$ yalnız sürekliyse ($f''$ yok) yine düzgün süreklilikten hata $\to0$ olur ama hız garantisi yoktur; $f$ süreksizse en büyük hata hiç sıfıra gitmez.
:::

### Derinlik neden? {#derinlik}

Teorem tek bir gizli katmanın **yettiğini** söylüyorsa neden derin ağlar? Çünkü bazı fonksiyonlar derinlikle **üstel olarak** daha az nöronla ifade edilir.

:::türetim Çadır haritası: 2L nöron, 2ᴸ parça
$g(x)=2\,\mathrm{ReLU}(x)-4\,\mathrm{ReLU}(x-\tfrac12)$, $[0,1]$ üzerinde: $x\le\frac12$ için $2x$, $x>\frac12$ için $2-2x$ — bir çadır. İki ReLU'lu tek bir katmandır. $g$, $[0,1]$'i kendine iki kez "katlayarak" eşler; bileşke $g^{(L)}=g\circ\dots\circ g$ ($L$ kez) her katlamada diş sayısını ikiye katlar: **$2^L$ doğrusal parça** (ve $2^L-1$ iç kırılma noktası), yalnız $2L$ nöron ve $L$ katmanla.

Tek gizli katmanlı bir ReLU ağı tek boyutlu girdide $\sum_k c_k\mathrm{ReLU}(a_kx+b_k)$ biçimindedir; her terim en fazla **bir** kırılma noktası ($x=-b_k/a_k$) ekler. $2^L-1$ kırılma noktası için en az $2^L-1$ nöron gerekir. Telgarsky (2016) bunun sağlam bir hâlini kanıtladı: derin ağların ifade edebildiği bazı fonksiyonları sığ ağlar ancak üstel genişlikle **yaklaşıklayabilir**.
:::

Pratik sonuç (ders slaytındaki kural): genellikle katman başına nöron sayısını artırmak yerine **katman sayısını** artırmak daha fazla getiri sağlar. Ama bu bir ifade gücü argümanıdır — derin ağı eğitmek kendi zorluklarını getirir (M3.5–M3.6, artık bağlantılar M4.2).

Aşağıdaki deneyde iki gösterim var: **genişlik** modunda hedef fonksiyonu $K$ menteşeyle yaklaşıkla ve hatanın $1/K^2$ ile azaldığını log–log grafikte gör (süreksiz basamakta azalmadığını da); **derinlik** modunda çadır haritasının bileşkesini ve gereken nöron sayılarını karşılaştır.

::lab l2

### Regresyon ve sınıflandırma için MLP tasarımı {#tasarim}

**Ağın girdileri:** elle çıkarılmış öznitelikler ya da ham veri — ikinci durumda ilk katmanlar öznitelik çıkarıcı gibi davranır. **Çıktıları:** sınıflandırmada etiketler (ikili {0,1}; çok sınıflı one-hot [1,0,…,0]; çok etiketli [1,0,1]…), regresyonda kestirimler. Çıktı katmanının tasarımı = çıktı düğümü sayısı + çıktı aktivasyonu (olasılık dağılımı mı, değil mi?) + kayıp.

**Regresyon.** Tahmin edilecek her değişken için bir çıktı nöronu. Çıktı nöronlarına genelde **aktivasyon uygulanmaz** (her değeri üretebilsinler). Çıktının hep pozitif olması gerekiyorsa ReLU ya da onun pürüzsüz versiyonu **softplus** $\log(1+e^x)$; belli bir aralıkta kalması gerekiyorsa lojistik ya da tanh (etiketleri o aralığa ölçekle). Kayıp tipik olarak MSE; aykırı değer çoksa MAE ya da Huber (M1.1).

| hiperparametre | regresyon için tipik değer |
|---|---|
| girdi nöronu | öznitelik başına bir |
| gizli katman | probleme bağlı, tipik olarak 1–5 |
| katman başına nöron | probleme bağlı, tipik olarak 10–100 |
| çıktı nöronu | tahmin boyutu başına bir |
| gizli aktivasyon | ReLU (ya da SELU) |
| çıktı aktivasyonu | yok; pozitifse ReLU/softplus; sınırlıysa lojistik/tanh |
| kayıp | MSE; aykırı değer varsa MAE/Huber |

**Sınıflandırma.**
- **İkili:** tek çıktı nöronu, lojistik aktivasyon; çıktı pozitif sınıfın tahmini olasılığıdır.
- **Çok etiketli ikili:** her pozitif etiket için bir lojistik çıktı. Örneğin bir e-posta için "spam mı?" ve "acil mi?" iki ayrı nöron. Olasılıkların toplamı 1 olmak **zorunda değildir**: her etiket kombinasyonu mümkündür.
- **Çok sınıflı** (her örnek üç ya da daha çok sınıftan yalnız birine ait; ör. 0–9 rakamları): sınıf başına bir çıktı nöronu ve tüm çıktı katmanına **softmax** $\mathrm{softmax}(z)_i=e^{z_i}/\sum_je^{z_j}$ (M1.5) — olasılıklar 0 ile 1 arasında ve toplamları 1 (sınıflar ayrık olduğu için gerekli). Üstel almanın tehlikesi (slayttaki soru): büyük $z$'de taşma — log-sum-exp hilesiyle çözülür (M1.5).
- **Kayıp:** çapraz entropi. İki olasılık kütle fonksiyonu için $H(p,q)=\sum_xp(x)\log\frac1{q(x)}=-\sum_xp(x)\log q(x)$; one-hot etiket $y$ gerçek dağılım, softmax çıktısı $\hat y$ kestirim: $\text{kayıp}=-\sum_iy_i\log\hat y_i$ (M0.4: NLL).

| | ikili | çok etiketli | çok sınıflı |
|---|---|---|---|
| çıktı nöronu | 1 | etiket başına 1 | sınıf başına 1 |
| çıktı aktivasyonu | sigmoid | sigmoid | softmax |
| kayıp | çapraz entropi | çapraz entropi (etiket başına) | çapraz entropi |
| PyTorch | `BCEWithLogitsLoss` | `BCEWithLogitsLoss` | `CrossEntropyLoss` (logit alır) |

Girdi, gizli katman sayısı ve genişlik, gizli aktivasyon üç durumda da regresyondakiyle aynı kurallardır. PyTorch'un kayıpları aktivasyonu **içerir** (logit alır): son katmana sigmoid/softmax koyup bir de bu kayıpları kullanmak çift uygulamadır.

**Bir problem için derin öğrenme:** (1) Derin öğrenmeye gerçekten ihtiyacın olduğundan emin ol. (2) Bir ağ modeli seç (kendin tasarlamak çok zordur): probleme özgü mimari; eğitim ve çıkarım için bellek ve hesap gereksinimi. (3) Veri topla, yöntemi belirle: gözetimli/gözetimsiz/yarı gözetimli, etiketleme, kayıp ve düzenlileştirme, hiperparametre ayarı (M3.9). (4) Test, dağıtım ve bakım: test performansı, köşe durumları, sürekli geliştirme.

### Derin ağ örnekleri {#ornekler}

| ağ (yıl) | tür | derinlik | parametre | getirdiği |
|---|---|---|---|---|
| LeNet-5 (1998) | CNN | 5 öğrenilen katman | ≈ 60 bin | ilk başarılı CNN'lerden; el yazısı rakamlar |
| AlexNet (2012) | CNN | 8 | ≈ 60 milyon | her şeyi başlatan model: GPU, ReLU, dropout |
| VGG-16 / VGG-19 (2014) | CNN | 16 / 19 | ≈ 138 / 144 milyon | yalnız 3×3 konvolüsyon yığını |
| GoogLeNet (2014) | CNN | 22 | ≈ 7 milyon | Inception bloğu, 1×1 konvolüsyon: daha az hesapla iyi performans |
| ResNet-34/50/101/152 (2015) | CNN | 34–152 (1202'ye kadar) | 21.8 / 25.6 / 44.5 / 60.2 milyon | artık bağlantı: çok derin ağlar eğitilebilir oldu |
| GPT-3 (2020) | Transformer | 96 dikkat katmanı | 175 milyar | büyük dil modeli |

:::uyarı Kaynak düzeltmeleri (giriş slaytları)
- **Görüntü sayısı:** slaytta 200×200 RGB görüntülerin sayısı $40\,000\times256^3\approx6.7\cdot10^{11}$ olarak verilmiş. Bu, "bir piksel konumu ve renk" çiftlerinin sayısıdır; görüntü sayısı her pikselin rengi **bağımsız** seçildiği için çarpımla değil üsle büyür: $(256^3)^{40\,000}=2^{960\,000}$ (yukarıdaki örnek).
- **VGG:** slaytta "VGG19 138 milyon parametre" deniyor; 138 milyon **VGG-16**'nın sayısıdır, VGG-19 ≈ 144 milyondur.
- **MLE:** "tüm tutarlı kestiriciler arasında en düşük ortalama kare hata" ifadesi fazla güçlü: MLE (düzenlilik koşulları altında) **asimptotik olarak verimlidir** — örnek sayısı sonsuza giderken varyansı Cramér–Rao alt sınırına ulaşır. Sonlu örnekte daha düşük MSE'li (yanlı) kestiriciler olabilir (ör. ridge, M1.3).
- **PReLU:** eğim $a$ bir hiperparametre değil, geri yayılımla **öğrenilen** bir parametredir (M3.5).
:::

**Neden şimdi?** Derin ağlar bu kadar iyiyse neden daha önce ve daha yaygın kullanılmadılar? Üç büyük engel vardı, bugün neredeyse hepsi aşıldı: **nasıl eğitilir?** (geri yayılım, ResNet'in artık bağlantıları, otomatik türev ve derin öğrenme API'leri); **eğitim ve çıkarım için donanım/yazılım** (Moore yasası, GPU'lar, CUDA); **eğitim verisi** (fonlama, web ve sosyal medya, daha hızlı ve büyük diskler, gözetimsiz/öz-gözetimli eğitim, M3.9).

### Çift iniş {#cift}

Klasik istatistik şunu söyler (M1.3): model karmaşıklığı arttıkça yanlılık azalır, varyans artar; test hatası **U** biçimlidir ve en iyi model ikisi arasındaki dengededir. Bu tablo, eğitim verisini **tam ezberleyen** (eğitim hatası 0) aşırı parametreli modellerin kötü genelleyeceğini öngörür. Oysa modern derin ağlar parametre sayısı veri sayısını çok aşar, eğitim verisini ezberler ve yine de iyi genelleşir.

**Çift iniş** (Belkin ve ark. 2019) bu iki gözlemi birleştirir: karmaşıklık ekseninde test hatası önce klasik U'yu çizer, **ara değerleme eşiğinde** (parametre sayısı ≈ veri sayısı) tepe yapar, sonra **yeniden düşer** — bazen klasik bölgenin en iyisinin de altına. Nakkiran ve ark. (2019) aynı olguyu derin ağlarda hem model boyutuna hem eğitim süresine göre gösterdi.

:::türetim Neden p = n'de tepe? (en küçük kareler)
$n$ örnek, $p$ öznitelik, $\Phi\in\mathbb R^{n\times p}$, model $\hat y=\Phi w$.
- $p<n$: en küçük kareler çözümü tektir, $w=(\Phi^\top\Phi)^{-1}\Phi^\top y$.
- $p>n$: $\Phi w=y$'yi sağlayan **sonsuz** çözüm vardır (her biri eğitim verisini tam ara değerler). Gradyan inişi sıfırdan başlarsa (ya da $\lambda\to0$ ridge limiti) en küçük normlu olanı seçer: $w=\Phi^\top(\Phi\Phi^\top)^{-1}y$.
- $p=n$: $\Phi$ kare; tek ara değerleyen çözüm $w=\Phi^{-1}y$. Rastgele bir kare matrisin en küçük tekil değeri çoğu zaman çok küçüktür; $\|w\|\le\|y\|/\sigma_{\min}(\Phi)$ ve gürültüyü ara değerlemek için ağırlıklar **patlar**. Test hatası, $\|w\|$ ile birlikte tepe yapar.

$p$, $n$'den uzaklaştıkça $\Phi\Phi^\top$ iyi koşullu olur ve en küçük normlu çözüm gürültüyü birçok yöne yayarak zararsızlaştırır. Ridge cezası ($\lambda>0$) SVD'de her tekil yöndeki $1/\sigma_i$ çarpanını $\sigma_i/(\sigma_i^2+\lambda)\le1/(2\sqrt\lambda)$ ile değiştirir: küçük tekil değerler artık ağırlıkları patlatamaz; uygun $\lambda$ tepeyi büyük ölçüde siler (Nakkiran ve ark. 2021).
:::

Aşağıdaki deney Hastie ve ark.'nın (2022) **gizli faktör modelini** kullanır: veri 5 boyutlu gizli bir $z$'den üretilir, her öznitelik $z$'nin gürültülü bir karışımıdır, hedef de $z$'ye bağlıdır. Model ilk $p$ özniteliği kullanır. Tepe $p=n$'de; aşırı parametreli bölge bu modelde klasik bölgenin en iyisinden daha iyi.

::lab l3

Ders slaytlarındaki kural "modern yapay sinir ağları düşük yanlılıklı modellerdir, performans sorunları genelde varyansla ilgilidir" bu tablonun pratik özetidir. M3.9'da yanlılık–varyans teşhisine ve hiperparametre seçimine dönüyoruz.

### Özet {#ozet}

- McCulloch–Pitts nöronu ikili girdilerle mantık hesaplar; TLU ağırlıklı toplam + basamak fonksiyonudur ve onu genelleştirir. Perceptron, tam bağlı tek bir TLU katmanıdır: $h=\varphi(XW+b)$.
- Perceptron kuralı (hata yapınca $w\leftarrow w+\eta yx$) perceptron kriterinin SGD'sidir; $w_0=0$ iken $\eta$ önemsizdir.
- Novikoff: $\gamma$ marjlı ayrılabilir veride en fazla $(R/\gamma)^2$ hata; ayrılamaz veride yakınsama yok.
- XOR doğrusal ayrılamaz (4 eşitsizlik çelişir); öznitelik dönüşümü ya da bir gizli katman çözer — gizli katman dönüşümü öğrenir.
- Aktivasyonsuz katmanlar tek bir doğrusal katmana çöker; evrensel yaklaşım teoremi tek gizli katmanın sürekli fonksiyonlar için **yettiğini**, ReLU menteşe inşası hatanın $\max|f''|/(8K^2)$ olduğunu, çadır haritası derinliğin üstel tasarruf sağlayabildiğini gösterir.
- Çıktı tasarımı: regresyonda aktivasyonsuz + MSE; ikili/çok etiketli sigmoid + BCE; çok sınıflı softmax + CE.
- Çift iniş: ara değerleme eşiğinde ($p\approx n$) test hatası tepe yapar, aşırı parametreli bölgede yeniden düşer; uygun düzenlileştirme tepeyi siler.
