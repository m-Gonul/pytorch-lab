### Neden bloklar? Aynı bütçeyle daha iyi önsel {#neden}

M4.1–M4.2 konvolüsyonun iki varsayımını kurdu: **yerellik** (bir piksel yalnız komşularıyla ilişkilidir) ve **ağırlık paylaşımı** (aynı desen görüntünün her yerinde aynı filtreyle aranır). Bu modül, bu iki önselin ötesinde CNN mimarisinin 2014–2020 arasında nasıl inceltildiğini **maliyet hesabıyla** anlatıyor: her yeni blok, belirli bir hesap/parametre bütçesinde daha fazla temsil gücü elde etmenin bir yoludur.

Önce önselin değerini ölçen ders alıştırması (Fashion-MNIST, 28×28):

| model | yapı | parametre |
|---|---|---|
| (2) CNN | 7×7×16 → havuz → 3×3×32 → havuz → 3×3×64 → havuz → FC 64 → 10 ("same" dolgu) | 61 514 |
| (3) FC | 784 → 74 → 50 → 10 | 62 350 |

:::türetim İki modelin parametreleri
Konvolüsyon katmanı $K^2C_{in}C_{out}+C_{out}$ (bias). CNN: $7^2\cdot1\cdot16+16=800$; $3^2\cdot16\cdot32+32=4\,640$; $3^2\cdot32\cdot64+64=18\,496$. Üç 2×2 havuzlama 28 → 14 → 7 → 3 ("same" konvolüsyon boyutu korur, havuzlama tabanı alır): düzleştirilmiş vektör $3\cdot3\cdot64=576$. FC 576 → 64: $36\,928$; 64 → 10: $650$. Toplam $61\,514$.

FC: $784\cdot74+74=58\,090$; $74\cdot50+50=3\,750$; $50\cdot10+10=510$. Toplam $62\,350$.
:::

Parametre bütçeleri neredeyse eşit; ama CNN'in parametrelerinin %60'ı tek bir FC katmanında, konvolüsyonların toplamı yalnız ≈ 24 bin. Yine de CNN belirgin biçimde daha iyi genelleşir (ve d aşamasında göreceğin gibi, rakamlar birkaç piksel kaydırılınca FC çöker, CNN dayanır). Nedeni kapasite değil **önseldir**: FC ağ 784 pikselin her birine ayrı ağırlık öğrenmek zorundadır; bir desenin sol üstte ve sağ altta aynı olduğunu her konum için ayrı ayrı veriden çıkarması gerekir. Konvolüsyon bunu yapı gereği bilir.

:::uyarı Parametre ≠ hesap ≠ süre
Bu modülde üç maliyet ayrı ayrı sayılır:
- **Parametre:** saklanan ağırlık sayısı. Konvolüsyonda $K^2C_{in}C_{out}/g$, görüntü boyutundan bağımsız.
- **Hesap:** çarpma–toplama (MAC) sayısı; FLOP ≈ 2 × MAC. Konvolüsyonda parametre × çıktı piksel sayısı ($H_{out}W_{out}$).
- **Süre:** donanımda geçen zaman. Hesaba ek olarak **bellek trafiğine** (ağırlıkları ve aktivasyonları okuyup yazmak) ve çekirdek başlatma gibi sabit maliyetlere bağlıdır. Hesabı az ama belleği çok okuyan katmanlar FLOP'larının düşündürdüğü kadar hızlı değildir (aşağıda, deney l1).
:::

### Inception modülü ve 1×1 darboğaz {#inception}

M4.2'de Inception modülünü ve 1×1 konvolüsyonların üç işlevini gördük. Burada hesabını yapıyoruz.

<figure class="fig wide"><svg viewBox="0 0 640 324" role="img" aria-label="Inception 3a modülü: dört paralel dal, 1×1 darboğazlar, çıktılar derinlik boyunca birleştirilir"><defs><marker id="i1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="230" y="250" width="180" height="26" rx="5" class="mute"/><text x="320" y="267" text-anchor="middle" class="sm">girdi 28×28×192</text><line x1="320" y1="250" x2="85" y2="222" class="ink"/><line x1="85" y1="222" x2="85" y2="160" class="ink"/><rect x="20" y="132" width="130" height="26" rx="5" class="soft-a ca"/><text x="85" y="149" text-anchor="middle" class="sm">1×1, 64</text><line x1="85" y1="222" x2="85" y2="160" class="ink" marker-end="url(#i1)"/><line x1="85" y1="132" x2="320" y2="86" class="ink" marker-end="url(#i1)"/><line x1="320" y1="250" x2="240" y2="222" class="ink"/><rect x="175" y="194" width="130" height="26" rx="5" class="soft-b cb"/><text x="240" y="211" text-anchor="middle" class="sm">1×1, 96</text><line x1="240" y1="222" x2="240" y2="222" class="ink" marker-end="url(#i1)"/><line x1="240" y1="194" x2="240" y2="160" class="ink" marker-end="url(#i1)"/><rect x="175" y="132" width="130" height="26" rx="5" class="soft-a ca"/><text x="240" y="149" text-anchor="middle" class="sm">3×3, 128</text><line x1="240" y1="132" x2="320" y2="86" class="ink" marker-end="url(#i1)"/><line x1="320" y1="250" x2="395" y2="222" class="ink"/><rect x="330" y="194" width="130" height="26" rx="5" class="soft-b cb"/><text x="395" y="211" text-anchor="middle" class="sm">1×1, 16</text><line x1="395" y1="222" x2="395" y2="222" class="ink" marker-end="url(#i1)"/><line x1="395" y1="194" x2="395" y2="160" class="ink" marker-end="url(#i1)"/><rect x="330" y="132" width="130" height="26" rx="5" class="soft-a ca"/><text x="395" y="149" text-anchor="middle" class="sm">5×5, 32</text><line x1="395" y1="132" x2="320" y2="86" class="ink" marker-end="url(#i1)"/><line x1="320" y1="250" x2="550" y2="222" class="ink"/><rect x="485" y="194" width="130" height="26" rx="5" class="mute"/><text x="550" y="211" text-anchor="middle" class="sm">3×3 maks. havuz</text><line x1="550" y1="222" x2="550" y2="222" class="ink" marker-end="url(#i1)"/><line x1="550" y1="194" x2="550" y2="160" class="ink" marker-end="url(#i1)"/><rect x="485" y="132" width="130" height="26" rx="5" class="soft-b cb"/><text x="550" y="149" text-anchor="middle" class="sm">1×1, 32</text><line x1="550" y1="132" x2="320" y2="86" class="ink" marker-end="url(#i1)"/><rect x="200" y="58" width="240" height="26" rx="5" class="soft-p cp"/><text x="320" y="75" text-anchor="middle" class="sm">derinlik boyunca birleştir: 64+128+32+32 = 256</text><text x="320" y="40" text-anchor="middle" class="sm">çıktı 28×28×256</text><text x="20" y="296" text-anchor="start" class="sm">Parametre (bias yok): 12 288 + 18 432 + 110 592 + 3 072 + 12 800 + 6 144 = 163 328 ≈ 159.5 Ki (makaledeki "159K").</text><text x="20" y="314" text-anchor="start" class="sm">Darboğazsız (192 kanala doğrudan 3×3 ve 5×5; havuz dalı 192 kanal): 387 072 parametre ve 416 kanallı çıktı.</text></svg><figcaption>Şekil 1 — GoogLeNet'in "inception (3a)" modülü (Szegedy ve ark. 2015, Tablo 1). Turuncu kutular 1×1 darboğazlar: pahalı 3×3 ve 5×5 konvolüsyonlardan önce kanal sayısını düşürür, havuz dalının kanalını da izdüşürür. Tüm dallar adım 1 ve "same" dolgu kullandığı için çıktılar aynı uzamsal boyuttadır ve kanal ekseninde birleştirilir.</figcaption></figure>

:::türetim 1×1 darboğazın kazancı
5×5 konvolüsyon, 192 → 32 kanal, bias yok: $25\cdot192\cdot32=153\,600$ parametre. Önce 1×1 ile 192 → 16, sonra 5×5 ile 16 → 32:
$$192\cdot16+25\cdot16\cdot32=3\,072+12\,800=15\,872,$$
yaklaşık **10 kat** az. 28×28'lik haritada MAC de aynı oranda düşer ($\times784$): 120 milyon yerine 12.4 milyon. Bedel: 5×5 filtre artık 192 kanalın tamamını değil, onların 16 boyutlu bir **doğrusal izdüşümünü** görür (ardından ReLU). Varsayım: kanallar arası bilgi düşük boyutlu bir alt uzayda yeterince temsil edilir. Bu bir **yaklaşıklık** (kısıt), eşitlik değil; kısıtın zararsız olduğunu deney (ImageNet doğruluğu) gösterir.
:::

Makalenin Tablo 1'indeki "inception (3a)" satırı "159K parametre, 128M işlem" der: $163\,328/1024\approx159.5$ ve $163\,328\cdot784\approx128.0$ milyon MAC. (Tablo "K"yı 1024 olarak kullanmış.) Darboğazsız aynı modül 387 072 parametre tutar ve havuz dalı 192 kanalı olduğu gibi geçirdiğinden çıktısı 416 kanala çıkar; modüller yığıldıkça kanal sayısı ve maliyet katlanarak büyürdü.

**Inception'ın sonraki sürümleri** (Szegedy ve ark. 2016, "Rethinking the Inception Architecture"):
- **5×5 → iki 3×3:** aynı alıcı alan, $25\to18$ ağırlık (%28 tasarruf; VGG'nin gerekçesiyle aynı, M4.2).
- **Asimetrik çarpanlama:** $n\times n\to1\times n$ ardından $n\times1$. 3×3 için $9\to6$ (%33); 7×7 için $49\to14$. Bu, çekirdeğin uzamsal olarak **ayrılabilir** olduğu varsayımıdır (aşağıda, deney l2).
- Batch normalizasyon (Inception-v2/BN-Inception), etiket düzleştirme (label smoothing, M3.7), yardımcı sınıflandırıcıların düzenlileştirici olarak yeniden yorumlanması.
- **Inception-v4 / Inception-ResNet** (2016): modülleri artık bağlantılarla birleştirir; topluluk modeliyle ImageNet top-5 hata ≈ %3.08.

### Ayrılabilir konvolüsyon: uzamsal ve derinlemesine {#ayrilabilir}

"Ayrılabilir konvolüsyon" iki farklı şey için kullanılır; ikisi de bir çarpanlama kısıtıdır.

**Uzamsal ayrılabilirlik.** Görüntü işlemede (özellikle donanımı kısıtlı gömülü sistemlerde) bilinen bir hiledir: dikey ve yatay çekirdeklerin ikisi de $[1,2,1]$ ise ardışık uygulanmalarının etkisi
$$\begin{bmatrix}1\\2\\1\end{bmatrix}\begin{bmatrix}1&2&1\end{bmatrix}=\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}$$
çekirdekli tek bir 2-B konvolüsyondur. Konvolüsyon birleşmeli olduğu için $(I*\mathbf u)*\mathbf v^\top=I*(\mathbf u\mathbf v^\top)$: $K^2$ yerine $2K$ çarpım.

:::türetim Hangi çekirdekler ayrılabilir? SVD
Bir $K\times K$ çekirdek $\mathbf M$ iki vektörün dış çarpımıysa ($\mathbf M=\mathbf u\mathbf v^\top$) rankı 1'dir; tersi de doğrudur. Genel bir çekirdek için tekil değer ayrışımı
$$\mathbf M=\sum_{i=1}^{K}\sigma_i\,\mathbf u_i\mathbf v_i^\top,\qquad\sigma_1\ge\sigma_2\ge\dots\ge0$$
onu rank-1 terimlerin toplamına ayırır; her terim bir dikey + bir yatay 1-B konvolüsyondur. Eckart–Young teoremi: ilk $r$ terim, Frobenius normunda **en iyi** rank-$r$ yaklaşıklıktır ve göreli hata $\sqrt{\sum_{i>r}\sigma_i^2/\sum_i\sigma_i^2}$'dir.
- Gauss: $e^{-(x^2+y^2)/2s^2}=e^{-x^2/2s^2}e^{-y^2/2s^2}$, rank 1. Kutu (ortalama) ve Sobel de rank 1.
- Gauss'un Laplace'ı: $\nabla^2G=G''(x)G(y)+G(x)G''(y)$, iki ayrılabilir terimin toplamı: rank 2.
- Disk ve öğrenilmiş filtreler: genelde yüksek rank; ayrılabilirlik bir kısıttır.
:::

::lab l2

**Derinlemesine ayrılabilir konvolüsyon** (depthwise separable) ise **kanal** ile **uzay** boyutlarını ayırır. Normal bir konvolüsyon katmanı, uzamsal desenleri (ör. bir oval) ve kanallar arası desenleri (ör. ağız + burun + gözler = yüz) **aynı anda** yakalamaya çalışan filtreler kullanır. Ayrılabilir katman ise uzamsal desenlerin ve kanallar arası desenlerin **ayrı ayrı modellenebileceği** gibi güçlü bir varsayım yapar ve iki parçadan oluşur:

1. **Derinlemesine (depthwise):** her girdi haritasına **tek bir** $K\times K$ uzamsal filtre (kanallar karışmaz). PyTorch: `nn.Conv2d(C, C, K, groups=C)`.
2. **Noktasal (pointwise):** yalnız kanallar arası desenlere bakan 1×1 bir konvolüsyon: `nn.Conv2d(C_in, C_out, 1)`.

:::türetim Maliyet oranı $1/C_{out}+1/K^2$
Çıktı pikseli başına MAC (ve bias'sız parametre): standart $K^2C_{in}C_{out}$; ayrılabilir $K^2C_{in}+C_{in}C_{out}$. Oran:
$$\frac{K^2C_{in}+C_{in}C_{out}}{K^2C_{in}C_{out}}=\frac{1}{C_{out}}+\frac{1}{K^2}.$$
$K=3$, $C_{out}=256$: $0.0039+0.111=0.115$, yani ≈ **8.7 kat** az. $C_{out}$ büyükken oran $\approx1/K^2$; büyük çekirdeklerde kazanç artar.
:::

<figure class="fig wide"><svg viewBox="0 0 640 256" role="img" aria-label="Standart konvolüsyon, Inception modülü ve derinlemesine ayrılabilir konvolüsyonun uzamsal ve kanal desenlerini nasıl ele aldığının karşılaştırması"><defs><marker id="i2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><text x="110" y="18" text-anchor="middle" class="sm">standart konvolüsyon</text><rect x="50" y="170" width="120" height="24" rx="5" class="mute"/><text x="110" y="186" text-anchor="middle" class="sm">girdi C<tspan baseline-shift="sub" font-size="9">in</tspan></text><line x1="110" y1="170" x2="110" y2="156" class="ink" marker-end="url(#i2)"/><rect x="20" y="130" width="180" height="26" rx="5" class="soft-a ca"/><text x="110" y="147" text-anchor="middle" class="sm">K×K, tüm kanallar birlikte</text><line x1="110" y1="130" x2="110" y2="52" class="ink" marker-end="url(#i2)"/><text x="110" y="46" text-anchor="middle" class="sm">çıktı C<tspan baseline-shift="sub" font-size="9">out</tspan></text><text x="110" y="216" text-anchor="middle" class="sm ta">uzay + kanal birlikte</text><text x="320" y="18" text-anchor="middle" class="sm">Inception (sadeleştirilmiş)</text><rect x="260" y="170" width="120" height="24" rx="5" class="mute"/><text x="320" y="186" text-anchor="middle" class="sm">girdi C<tspan baseline-shift="sub" font-size="9">in</tspan></text><line x1="320" y1="170" x2="320" y2="156" class="ink" marker-end="url(#i2)"/><rect x="230" y="130" width="180" height="26" rx="5" class="soft-b cb"/><text x="320" y="147" text-anchor="middle" class="sm">1×1 (yalnız kanal)</text><line x1="320" y1="130" x2="320" y2="106" class="ink" marker-end="url(#i2)"/><rect x="230" y="80" width="180" height="26" rx="5" class="soft-a ca"/><text x="320" y="97" text-anchor="middle" class="sm">3×3 (uzay + kanal)</text><line x1="320" y1="80" x2="320" y2="52" class="ink" marker-end="url(#i2)"/><text x="320" y="46" text-anchor="middle" class="sm">çıktı C<tspan baseline-shift="sub" font-size="9">out</tspan></text><text x="320" y="216" text-anchor="middle" class="sm ta">önce kanal, sonra ikisi</text><text x="530" y="18" text-anchor="middle" class="sm">derinlemesine ayrılabilir</text><rect x="470" y="170" width="120" height="24" rx="5" class="mute"/><text x="530" y="186" text-anchor="middle" class="sm">girdi C<tspan baseline-shift="sub" font-size="9">in</tspan></text><line x1="530" y1="170" x2="530" y2="156" class="ink" marker-end="url(#i2)"/><rect x="440" y="130" width="180" height="26" rx="5" class="soft-p cp"/><text x="530" y="147" text-anchor="middle" class="sm">K×K, kanal başına ayrı</text><line x1="530" y1="130" x2="530" y2="106" class="ink" marker-end="url(#i2)"/><rect x="440" y="80" width="180" height="26" rx="5" class="soft-b cb"/><text x="530" y="97" text-anchor="middle" class="sm">1×1 (yalnız kanal)</text><line x1="530" y1="80" x2="530" y2="52" class="ink" marker-end="url(#i2)"/><text x="530" y="46" text-anchor="middle" class="sm">çıktı C<tspan baseline-shift="sub" font-size="9">out</tspan></text><text x="530" y="216" text-anchor="middle" class="sm ta">uzay ve kanal ayrı ayrı</text><text x="20" y="246" text-anchor="start" class="sm">Parametre (3×3, C_in = 128, C_out = 256):  standart 294 912   ·   derinlemesine 1 152 + noktasal 32 768 = 33 920 (8.7 kat az)</text></svg><figcaption>Şekil 2 — Xception'ın ("Extreme Inception") bakışı. Standart konvolüsyon uzamsal desenleri (oval) ve kanallar arası desenleri (ağız + burun + göz = yüz) aynı filtrede yakalar. Inception modülü önce 1×1 ile yalnız kanallara bakar, sonra normal bir 3×3 uygular: ikisinin arası. Derinlemesine ayrılabilir katman ikisini tamamen ayırır: her girdi kanalına tek bir uzamsal filtre, sonra yalnız kanalları karıştıran 1×1.</figcaption></figure>

**Xception** ("Extreme Inception"; Chollet 2016): Inception-v4 gibi GoogLeNet ile ResNet fikirlerini birleştirir ama Inception modüllerini derinlemesine ayrılabilir katmanlarla değiştirir. Bu katmanlar daha önce de kullanılmıştı ama Xception'da olduğu kadar merkezî değildi. 350 milyon görüntü ve 17 000 sınıflı çok büyük bir görevde (Google'ın iç veri kümesi JFT) Inception-v3'ü belirgin biçimde geçti. Neden Inception'ın bir varyantı sayılır? Inception modülünde 1×1 katmanlar yalnız kanallara bakar, üstlerindeki 3×3/5×5 katmanlar ise hem uzaya hem kanallara: modül, normal konvolüsyonla (ikisi birlikte) ayrılabilir konvolüsyonun (ikisi ayrı) **arasında** bir yerdedir. Xception bu ayrımı uca götürür.

Ayrılabilir katmanlar her girdi kanalı için yalnız **bir** uzamsal filtreye sahip olduğundan, kanalı çok az olan katmanlardan (girdi katmanı gibi, 3 kanal) sonra kullanılmamalıdır. Bu yüzden Xception iki normal konvolüsyonla başlar, sonra yalnız ayrılabilir konvolüsyonlar (toplam 34), birkaç max havuzlama ve her zamanki son katmanlar (küresel ortalama havuzlama ve yoğun çıkış) gelir. Neredeyse tüm modüller doğrusal artık bağlantılıdır. Chollet ayrıca derinlemesine ve noktasal katmanlar **arasına aktivasyon koymamanın** daha iyi sonuç verdiğini bildirir (MobileNet'in aksine).

:::uyarı Kaynak notu: "varsayılan olarak ayrılabilir kullan"
Slayt (Géron'dan) ayrılabilir katmanların daha az parametre, bellek ve hesap kullandığını ve "genelde daha iyi" performans verdiğini, bu yüzden varsayılan olarak düşünülmesi gerektiğini söyler. Parametre ve MAC iddiaları kesindir (yukarıdaki türetim). "Daha iyi" ve "daha hızlı" ise koşula bağlıdır: (1) Doğruluk kazancı Xception'ın JFT ve ImageNet deneylerinde ölçüldü; küçük modellerde aynı FLOP bütçesinde standart konvolüsyon da iyi olabilir. (2) GPU'da derinlemesine katmanın **aritmetik yoğunluğu** çok düşüktür (her ağırlık piksel başına yalnız bir çarpmada kullanılır): süre bellek trafiğiyle sınırlanır, MAC'teki 9 katlık azalma sürede görünmeyebilir. Mobil işlemcilerde kazanç gerçektir; bu yüzden MobileNet ailesi orada yaygındır.
:::

### Grup konvolüsyonu ve ResNeXt {#grup}

**Grup konvolüsyonu** girdi kanallarını $g$ gruba böler; her çıktı grubu yalnız kendi girdi grubuna bağlıdır. Parametre ve MAC $g$'ye bölünür: $K^2C_{in}C_{out}/g$. Ağırlık matrisi (kanal ekseninde) **blok köşegendir**. Uç durumlar: $g=1$ normal konvolüsyon, $g=C_{in}=C_{out}$ derinlemesine konvolüsyon.

Kökeni bir mühendislik zorunluluğudur: AlexNet (2012) iki GPU'ya bölündü ve bazı katmanlar yalnız aynı GPU'daki haritalara bağlıydı ($g=2$). Krizhevsky ve ark. iki grubun farklı özelliklere (biri renkten bağımsız kenarlar, öteki renk lekeleri) uzmanlaştığını gözlemledi.

**ResNeXt** (Xie ve ark. 2017) bunu bir tasarım ilkesine çevirdi: darboğaz bloğundaki 3×3'ü $C$ paralel dala ("**kardinalite**") bölmek, bir grup konvolüsyonuna eşdeğerdir. ResNeXt-50 (32×4d) bloğu, 256 kanallı aşamada: 1×1 256 → 128, 3×3 grup konvolüsyonu 128 → 128 ($g=32$), 1×1 128 → 256:
$$256\cdot128+\frac{9\cdot128\cdot128}{32}+128\cdot256=32\,768+4\,608+32\,768=70\,144,$$
ResNet-50 bloğunun $256\cdot64+9\cdot64\cdot64+64\cdot256=69\,632$'sine çok yakın. Aynı bütçede daha geniş (128 kanallı) bir ara temsil ve ImageNet'te daha düşük hata. Bulgu: kardinaliteyi artırmak, derinlik ya da genişliği artırmaktan daha verimli olabilir.

Grupların sakıncası, bilginin gruplar arasında akmamasıdır; grup konvolüsyonundan sonra gelen 1×1 katman bunu karıştırır. **ShuffleNet** (Zhang ve ark. 2018) 1×1 katmanları da gruplayıp araya ucuz bir **kanal karıştırma** (channel shuffle: yeniden şekillendir, devrik al, düzleştir) koyar.

::lab l1

### Sıkıştır-uyar (SE) blokları {#se}

ILSVRC-2017'nin kazananı **SENet** oldu (Hu ve ark.; %2.25 top-5 hata). SENet yeni bir mimari değil, mevcutları (Inception ağları, ResNet'ler) genişleten bir **eklentidir**: SE-Inception ve SE-ResNet. Kazanç, özgün mimarideki her birime (her Inception modülüne ya da her artık birime) eklenen küçük bir sinir ağından, **SE bloğundan** gelir.

<figure class="fig wide"><svg viewBox="0 0 640 210" role="img" aria-label="Squeeze-and-excitation bloğu: küresel ortalama havuzlama, darboğazlı iki tam bağlı katman ve sigmoid; çıktı kanal başına ölçek olarak girdiyle çarpılır"><defs><marker id="i3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="10" y="90" width="110" height="40" rx="5" class="soft-a ca"/><text x="65" y="114" text-anchor="middle" class="sm">X: H×W×C</text><line x1="120" y1="110" x2="150" y2="110" class="ink" marker-end="url(#i3)"/><rect x="150" y="96" width="120" height="28" rx="5" class="soft-b cb"/><text x="210" y="114" text-anchor="middle" class="sm">GAP → 1×1×C</text><line x1="270" y1="110" x2="290" y2="110" class="ink" marker-end="url(#i3)"/><rect x="290" y="96" width="110" height="28" rx="5" class="soft-p cp"/><text x="345" y="114" text-anchor="middle" class="sm">FC C/r + ReLU</text><line x1="400" y1="110" x2="420" y2="110" class="ink" marker-end="url(#i3)"/><rect x="420" y="96" width="120" height="28" rx="5" class="soft-p cp"/><text x="480" y="114" text-anchor="middle" class="sm">FC C + sigmoid</text><text x="345" y="88" text-anchor="middle" class="sm">sıkıştır (squeeze)</text><text x="480" y="88" text-anchor="middle" class="sm">uyar (excitation)</text><line x1="540" y1="110" x2="575" y2="110" class="ink" marker-end="url(#i3)"/><circle cx="590" cy="110" r="12" class="ink" fill="none"/><text x="590" y="115" text-anchor="middle" class="sm">⊗</text><path d="M 65 130 V 170 H 590 V 124" class="ink" fill="none" marker-end="url(#i3)"/><line x1="590" y1="98" x2="590" y2="60" class="ink" marker-end="url(#i3)"/><text x="590" y="52" text-anchor="middle" class="sm">X̃ = s ⊙ X</text><text x="150" y="150" text-anchor="start" class="sm">s ∈ (0, 1)<tspan baseline-shift="super" font-size="9">C</tspan>: kanal başına bir ölçek</text><text x="10" y="200" text-anchor="start" class="sm">Parametre: 2C²/r (+ C/r + C bias). C = 256, r = 16: 8 464. ResNet-50'ye ≈ %10 parametre, ≈ %0.3 FLOP ekler.</text></svg><figcaption>Şekil 3 — Sıkıştır-uyar (SE) bloğu (Hu ve ark. 2018). Blok yalnız kanal boyutuna bakar: her haritanın ortalamasını alır, bunu C/r boyutlu bir gömmeye sıkıştırır, sonra her kanal için 0–1 arası bir yeniden ölçekleme katsayısı üretir. İlgisiz haritalar küçültülür, ilgililer olduğu gibi kalır. Bir artık birimde SE, artık dalın çıktısına toplamadan önce uygulanır.</figcaption></figure>

Bir SE bloğu bağlı olduğu birimin çıktısını yalnız **derinlik** boyutunda analiz eder (uzamsal desen aramaz) ve hangi özniteliklerin genellikle **birlikte** etkin olduğunu öğrenir. Sonra bu bilgiyle öznitelik haritalarını yeniden ayarlar. Örneğin bir SE bloğu ağızların, burunların ve gözlerin resimlerde genellikle birlikte göründüğünü öğrenebilir: ağız ve burun görürsen göz de görmeyi beklersin. Blok ağız ve burun haritalarında güçlü, göz haritasında yalnız hafif bir etkinlik görürse göz haritasını **güçlendirir** (daha doğrusu, ilgisiz haritaları **zayıflatır**). Gözler başka bir şeyle karışmışsa bu yeniden ayarlama belirsizliği çözmeye yardım eder.

:::tanım SE bloğu
Girdi $\mathbf X\in\mathbb R^{C\times H\times W}$, daraltma oranı $r$ (tipik 16).
1. **Sıkıştır (squeeze):** küresel ortalama havuzlama, $z_c=\frac{1}{HW}\sum_{i,j}X_{c,i,j}$. 256 haritalık bir girdi için 256 sayı: her filtrenin genel tepki düzeyi.
2. **Uyar (excitation):** $\mathbf s=\sigma\big(W_2\,\mathrm{ReLU}(W_1\mathbf z+\mathbf b_1)+\mathbf b_2\big)$, $W_1\in\mathbb R^{C/r\times C}$, $W_2\in\mathbb R^{C\times C/r}$. Gizli katman harita sayısından belirgin biçimde az nörona sahiptir (256 kanal için 16): 256 sayı küçük bir vektöre sıkıştırılır — öznitelik tepkilerinin dağılımının düşük boyutlu bir **gömmesi**. Bu darboğaz, bloğu öznitelik **birlikteliklerinin** genel bir temsilini öğrenmeye zorlar. Çıkış katmanı her harita için 0 ile 1 arası bir sayı verir.
3. **Yeniden ölçekle:** $\tilde{\mathbf X}_c=s_c\,\mathbf X_c$. Düşük skorlu ilgisiz öznitelikler küçülür, 1'e yakın skorlular olduğu gibi kalır.

Parametre: $2C^2/r+C/r+C$. $C=256$, $r=16$: $8\,464$.
:::

**Maliyet.** SE-ResNet-50, ResNet-50'nin ≈ 25 milyon parametresine ≈ 2.5 milyon (%10) ekler, ama hesaba yalnız ≈ %0.26: SE bloğu $H\times W$ haritada değil, $C$ boyutlu tek bir vektörde çalışır. Parametrelerin çoğu son aşamadaki ($C=2048$) bloklardadır.

:::sezgi SE bir kanal dikkatidir
$\mathbf s$ girdiye bağlı, 0–1 arası çarpımsal ağırlıklar üretir: bir **geçit** (LSTM'deki gibi, M4.4) ya da kanallar üzerinde bir dikkat. Fark: dikkat (M4.6) ağırlıkları softmax ile toplamı 1 olacak biçimde normalize eder ve konumlar arasında karıştırır; SE kanalları bağımsız sigmoid'lerle ölçekler ve hiçbir şeyi karıştırmaz. CBAM gibi sonraki bloklar buna bir uzamsal dikkat haritası ekler; EfficientNet ve MobileNetV3 SE'yi standart parça olarak kullanır.
:::

### Küresel ortalama havuzlama ve sınıf aktivasyon haritaları {#gap}

**Küresel ortalama havuzlama** (GAP; Lin ve ark. 2013, "Network in Network") her öznitelik haritasının ortalamasını alır: $C\times H\times W\to C$. GoogLeNet'ten beri sınıflandırma ağlarının son katmanı GAP + tek bir doğrusal katmandır; düzleştir + büyük FC katmanlarının (VGG-16'da 100 milyonun üzerinde parametre) yerini alır. Kazançları: çok daha az parametre ve aşırı uyum, **her girdi boyutunu** kabul etme (GAP her $H\times W$'yi tek sayıya indirger), ve aşağıdaki yorumlanabilirlik.

:::türetim Logit = CAM'in ortalaması
Son konvolüsyon haritaları $f_k(x,y)$, GAP + doğrusal katmanın sınıf $c$ ağırlıkları $w_{ck}$:
$$\text{logit}_c=\sum_k w_{ck}\frac{1}{HW}\sum_{x,y}f_k(x,y)+b_c=\frac{1}{HW}\sum_{x,y}\underbrace{\sum_kw_{ck}f_k(x,y)}_{\mathrm{CAM}_c(x,y)}+b_c.$$
Toplamların yer değiştirmesi; GAP ve doğrusal katman doğrusal olduğu için **kesin eşitlik**. $\mathrm{CAM}_c(x,y)$ her konumun $c$ sınıfının skoruna katkısıdır (Zhou ve ark. 2016). Ağ yalnız görüntü etiketleriyle eğitildiği hâlde harita nesnenin bulunduğu yerde parlar: **zayıf denetimli yerelleştirme**.
:::

Koşullar ve sınırlar: son haritalar düşük çözünürlüklüdür (224 girdide 7×7), CAM büyütülerek kaba bir yerelleştirme verir; ve yalnız "GAP + tek doğrusal katman" yapısı için tanımlıdır. **Grad-CAM** (Selvaraju ve ark. 2017) kanal ağırlıklarını $w_{ck}$ yerine logit'in haritalara göre gradyanının uzamsal ortalamasıyla tanımlayarak her mimariye genelleştirir (GAP + doğrusal katmanda ikisi orantılıdır).

::lab l3

### DenseNet: toplamak yerine birleştirmek {#densenet}

**DenseNet** (Huang ve ark. 2017) bir "yoğun blok" içinde her katmanı **önceki tüm katmanlara** bağlar, ama ResNet gibi toplayarak değil, **kanal ekseninde birleştirerek**: $\mathbf x_l=H_l([\mathbf x_0,\mathbf x_1,\dots,\mathbf x_{l-1}])$. Her katman yalnız $k$ yeni harita ekler (**büyüme oranı**, tipik $k=32$); $l$. katmanın girdisi $k_0+k(l-1)$ kanaldır. Katmanlar dar olabilir çünkü önceki tüm öznitelikleri doğrudan görürler; öznitelik yeniden kullanımı parametre verimliliği sağlar (DenseNet-121 ≈ 8 milyon parametre). Girdi kanalı doğrusal büyüdüğünden her katmana önce bir 1×1 darboğaz ($4k$ kanala; DenseNet-B), bloklar arasına da kanalı yarıya indiren geçiş katmanları (1×1 + 2×2 ortalama havuz; sıkıştırma 0.5) konur. Bedel bellektir: birleştirilen haritalar geri geçiş için saklanır (paylaşımlı bellekli uygulamalar bunu azaltır).

### MobileNet ailesi ve ters artık blok {#mobilenet}

**MobileNetV1** (Howard ve ark. 2017): neredeyse tamamen derinlemesine ayrılabilir katmanlardan (her ikisinin ardından BN + ReLU) oluşan bir ağ. İki küresel düğme: **genişlik çarpanı** $\alpha$ her katmanın kanal sayısını çarpar, **çözünürlük çarpanı** $\rho$ girdi boyutunu. Maliyet $\approx\alpha^2\rho^2$ ile ölçeklenir (kanal karışımı $C_{in}C_{out}\propto\alpha^2$, piksel sayısı $\propto\rho^2$).

<figure class="fig wide"><svg viewBox="0 0 640 240" role="img" aria-label="ResNet darboğaz bloğunda kanal sayısı ortada daralır; MobileNetV2 ters artık bloğunda ortada genişler ve son katman doğrusaldır"><defs><marker id="i4" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><text x="130" y="18" text-anchor="middle" class="sm">ResNet darboğazı (geniş → dar → geniş)</text><rect x="30" y="170" width="200" height="26" rx="5" class="mute"/><text x="130" y="187" text-anchor="middle" class="sm"></text><text x="130" y="187" text-anchor="middle" class="sm">girdi 256</text><rect x="95" y="120" width="70" height="26" rx="5" class="soft-b cb"/><text x="130" y="137" text-anchor="middle" class="sm"></text><text x="130" y="137" text-anchor="middle" class="sm">1×1 → 64</text><line x1="130" y1="170" x2="130" y2="148" class="ink" marker-end="url(#i4)"/><rect x="95" y="70" width="70" height="26" rx="5" class="soft-a ca"/><text x="130" y="87" text-anchor="middle" class="sm"></text><text x="130" y="87" text-anchor="middle" class="sm">3×3, 64</text><line x1="130" y1="120" x2="130" y2="98" class="ink" marker-end="url(#i4)"/><rect x="30" y="20" width="200" height="26" rx="5" class="soft-b cb"/><text x="130" y="37" text-anchor="middle" class="sm"></text><text x="130" y="37" text-anchor="middle" class="sm">1×1 → 256</text><line x1="130" y1="70" x2="130" y2="48" class="ink" marker-end="url(#i4)"/><path d="M 234 183 H 246 V 33 H 236" class="cq dash" fill="none" marker-end="url(#i4)"/><text x="130" y="222" text-anchor="middle" class="sm">pahalı 3×3 dar tarafta</text><text x="450" y="18" text-anchor="middle" class="sm">MobileNetV2 ters artığı (dar → geniş → dar)</text><rect x="410" y="170" width="80" height="26" rx="5" class="mute"/><text x="450" y="187" text-anchor="middle" class="sm"></text><text x="450" y="187" text-anchor="middle" class="sm">girdi 24</text><rect x="345" y="120" width="210" height="26" rx="5" class="soft-b cb"/><text x="450" y="137" text-anchor="middle" class="sm"></text><text x="450" y="137" text-anchor="middle" class="sm">1×1 → 144, ReLU6</text><line x1="450" y1="170" x2="450" y2="148" class="ink" marker-end="url(#i4)"/><rect x="345" y="70" width="210" height="26" rx="5" class="soft-p cp"/><text x="450" y="87" text-anchor="middle" class="sm"></text><text x="450" y="87" text-anchor="middle" class="sm">3×3 derinlemesine, ReLU6</text><line x1="450" y1="120" x2="450" y2="98" class="ink" marker-end="url(#i4)"/><rect x="410" y="20" width="80" height="26" rx="5" class="soft-b cb"/><text x="450" y="37" text-anchor="middle" class="sm"></text><text x="450" y="37" text-anchor="middle" class="sm">1×1 → 24</text><line x1="450" y1="70" x2="450" y2="48" class="ink" marker-end="url(#i4)"/><path d="M 559 183 H 571 V 33 H 496" class="cq dash" fill="none" marker-end="url(#i4)"/><text x="450" y="222" text-anchor="middle" class="sm">kısayol dar uçları bağlar; son 1×1 doğrusal</text></svg><figcaption>Şekil 4 — Kutuların genişliği kanal sayısını gösteriyor. ResNet darboğazı pahalı 3×3 konvolüsyonu dar tutar; kısayol geniş uçlar arasındadır. MobileNetV2 tersini yapar: dar bir "darboğaz" temsilinden 1×1 ile t = 6 kat genişler, ucuz derinlemesine 3×3 uygular, aktivasyonsuz (doğrusal) bir 1×1 ile tekrar daralır; kısayol dar uçları bağlar.</figcaption></figure>

**MobileNetV2** (Sandler ve ark. 2018) **ters artık** (inverted residual) blok ve **doğrusal darboğaz** önerir:
1. 1×1 ile dar girdiyi $t$ kat genişlet (tipik $t=6$), ReLU6.
2. Geniş temsil üzerinde derinlemesine 3×3, ReLU6.
3. 1×1 ile tekrar dar boyuta izdüşür — **aktivasyon yok**.
4. Girdi ve çıktı aynı boyuttaysa (adım 1) kısayol **dar uçları** bağlar.

ResNet darboğazının tersidir: orada geniş uçlar arasında dar bir orta katman vardı. Gerekçe: derinlemesine konvolüsyon ucuz olduğundan geniş bir temsil üzerinde uygulanabilir; bellekte ise (kısayol ve bloklar arası aktivasyonlar) yalnız dar tensörler taşınır. **Doğrusal darboğaz:** düşük boyutlu bir temsile ReLU uygulamak bilgiyi geri dönüşsüz siler (negatif yarı uzayın tamamı sıfıra çöker). Sandler ve ark. düşük boyutlu bir manifoldu rastgele bir matrisle yüksek boyuta gömüp ReLU ve tersini uyguladıklarında, ara boyut küçükken manifoldun çöktüğünü gösterir; deneyde de son katmandaki ReLU'yu kaldırmak doğruluğu artırır. ReLU6 ($\min(\max(0,z),6)$) düşük duyarlıklı (8 bit) hesapta sınırlı aralık sağlamak içindir.

:::uyarı Ters artık kendiliğinden ucuz değildir
Aynı $C_{in}=C_{out}=32$ için standart 3×3 konvolüsyon $9\cdot32\cdot32=9\,216$ parametre; $t=6$ ters artık $32\cdot192+9\cdot192+192\cdot32=14\,016$. Kazanç, **192 kanallı** bir temsili işlemeyi ucuzlatmasındadır: aynı genişlikte standart bir 3×3, $9\cdot192\cdot192=331\,776$ parametre tutardı. Deney l1'deki "MobileNetV2 tipik blok" ayarına bak.
:::

### EfficientNet: bileşik ölçekleme {#efficientnet}

Bir ağı büyütmenin üç yolu var: **derinlik** $d$ (katman sayısı), **genişlik** $w$ (kanal sayısı), **çözünürlük** $r$ (girdi boyutu). Tek bir boyutu büyütmek bir noktadan sonra azalan getiri verir: çok derin ağlar eğitilmesi zor, çok geniş ama sığ ağlar üst düzey öznitelikleri yakalamakta zorlanır, çok yüksek çözünürlükte alıcı alan yetmez. Tan ve Le (2019) üçünü **birlikte** ölçekler:

:::türetim Bileşik ölçekleme kuralı
Bir katmanın FLOP'u derinlikle doğrusal, genişlikle karesel ($C_{in}C_{out}$), çözünürlükle karesel ($HW$) büyür:
$$\text{FLOP}\propto d\cdot w^2\cdot r^2.$$
Ölçek çarpanları $d=\alpha^\phi$, $w=\beta^\phi$, $r=\gamma^\phi$ ve kısıt $\alpha\beta^2\gamma^2\approx2$ seçilirse FLOP $\approx2^\phi$ ile büyür: $\phi$ bir birim artınca hesap iki katına çıkar. $\alpha,\beta,\gamma$ küçük temel ağda ($\phi=1$) küçük bir ızgara aramasıyla bulunur: $\alpha=1.2$, $\beta=1.1$, $\gamma=1.15$; $1.2\cdot1.1^2\cdot1.15^2=1.92\approx2$. Sonra $\phi$ büyütülerek B1…B7 elde edilir. (Kural bir **sezgisel**: sabit oranların her ölçekte en iyi olduğunu varsayar; makale bunu yalnız deneyle destekler.)
:::

Temel ağ (EfficientNet-B0) bir mimari aramasıyla (NAS) bulundu ve MobileNetV2'nin ters artık bloklarına SE ekler (MBConv). EfficientNet-B7 o dönemin en iyi ImageNet doğruluğuna yakın sonucu çok daha az parametreyle elde etti. Sonraki çalışmalar (EfficientNetV2, RegNet, ConvNeXt) derinlemesine katmanların GPU'daki yavaşlığına karşı **eğitim süresini** de hedefe kattı: ilk aşamalarda derinlemesine katmanları yeniden normal konvolüsyonla değiştirmek (Fused-MBConv) gibi.

:::ayrıntı ConvNeXt: CNN'in Transformer'dan öğrendikleri
Liu ve ark. (2022) bir ResNet-50'yi adım adım Vision Transformer'ın (M4.15) tasarım seçimlerine yaklaştırdı: 7×7 derinlemesine konvolüsyon (büyük alıcı alan, ucuz), ters darboğaz (genişleme 4), BN yerine LayerNorm, ReLU yerine GELU, daha az normalizasyon ve aktivasyon, "patchify" ilk katman (4×4, adım 4). Sonuç, aynı hesapta Swin Transformer kadar iyi saf bir CNN oldu. Ders: mimari başarının önemli bir kısmı blok tasarımından ve eğitim tarifinden gelir, "konvolüsyon mu dikkat mi" sorusundan değil.
:::

### PyTorch'ta {#pytorch}

| blok | PyTorch |
|---|---|
| grup / derinlemesine | `nn.Conv2d(Cin, Cout, K, padding=K//2, groups=g)`; derinlemesine: `groups=Cin`, `Cout = Cin` (ya da katı) |
| noktasal | `nn.Conv2d(Cin, Cout, 1)` |
| ayrılabilir blok | `nn.Sequential(dw, nn.BatchNorm2d(C), nn.ReLU(), pw, nn.BatchNorm2d(Cout), nn.ReLU())` |
| GAP | `nn.AdaptiveAvgPool2d(1)` + `torch.flatten(x, 1)` ya da `x.mean((2, 3))` |
| SE | `s = torch.sigmoid(fc2(F.relu(fc1(x.mean((2, 3))))))`; `x * s[:, :, None, None]`; hazır: `torchvision.ops.SqueezeExcitation` |
| kanal karıştırma | `x.view(B, g, C // g, H, W).transpose(1, 2).reshape(B, C, H, W)` |
| hazır modeller | `torchvision.models.mobilenet_v2`, `efficientnet_b0`, `resnext50_32x4d`, `densenet121`, `convnext_tiny` (`weights=...DEFAULT`) |
| sayım | `sum(p.numel() for p in m.parameters())`; FLOP için `torch.utils.flop_counter.FlopCounterMode` |
| süre | `torch.cuda.synchronize()` ile çevrelenmiş ısınmalı ölçüm ya da `torch.utils.benchmark.Timer` |

### Özet {#ozet}

- Eşit parametreli FC ve CNN'de CNN'in üstünlüğü kapasiteden değil önselden gelir (yerellik, ağırlık paylaşımı).
- 1×1 darboğaz, pahalı konvolüsyonun önünde kanalı daraltır: Inception (3a)'da 5×5 dal ≈ 10 kat ucuzlar. Inception-v3: 5×5 → iki 3×3, $n\times n\to1\times n+n\times1$.
- Uzamsal ayrılabilirlik rank-1 çekirdek demektir (SVD ile test edilir). Derinlemesine ayrılabilir konvolüsyon uzay ve kanalı ayırır: maliyet oranı $1/C_{out}+1/K^2$. Xception bunu uca götürür; az kanallı katmanlardan sonra kullanılmaz.
- Grup konvolüsyonu maliyeti $g$'ye böler; ResNeXt'in kardinalitesi; ShuffleNet'in kanal karıştırması.
- SE: GAP → FC $C/r$ → ReLU → FC $C$ → sigmoid → kanal başına ölçek; %10 parametre, ≈ %0.3 hesap.
- GAP + doğrusal katmanda logit = CAM'in ortalaması: etiketlerle eğitilen ağ nesneyi yerelleştirir.
- MobileNetV2: dar → geniş derinlemesine → dar, son 1×1 doğrusal. EfficientNet: $d\cdot w^2\cdot r^2$ kuralıyla bileşik ölçekleme.
- FLOP ≠ süre: düşük aritmetik yoğunluklu katmanlar (derinlemesine) bellekle sınırlıdır.
