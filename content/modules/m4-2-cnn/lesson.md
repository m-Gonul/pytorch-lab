### Mimarinin kalıbı {#kalip}

M4.1'deki yapı taşları — konvolüsyon, ReLU, havuzlama, sonda tam bağlı katmanlar — bir CNN'in iskeletidir: birkaç konvolüsyon (+ReLU), bir havuzlama, yine birkaç konvolüsyon, yine havuzlama… Görüntü ağ boyunca **küçülür** ama **derinleşir** (daha çok öznitelik haritası); tepede birkaç tam bağlı katman ve tahmini veren son katman (ör. softmax). Bu modülde bu kalıbın yıllar içinde nasıl evrildiğini, her adımda hangi problemin çözüldüğünü izliyoruz: LeNet-5 → AlexNet → VGG → GoogLeNet → ResNet. Her mimari yeni bir fikir getirdi; bu fikirlerin çoğu bugünkü ağlarda hâlâ duruyor.

| ağ (yıl) | ImageNet top-5 hata | derinlik | parametre | getirdiği |
|---|---|---|---|---|
| LeNet-5 (1998) | — (MNIST) | 7 katman | ≈ 60 bin | konvolüsyon + havuzlama + gradyanla uçtan uca eğitim |
| AlexNet (2012) | %15.3 (ILSVRC-2012, topluluk) | 8 | ≈ 60 milyon | GPU, ReLU, dropout, veri artırma, ardışık konvolüsyonlar |
| VGG-16/19 (2014) | %7.3 (topluluk) | 16 / 19 | ≈ 138 / 144 milyon | yalnız 3×3 konvolüsyon yığını |
| GoogLeNet (2014) | %6.7 | 22 | ≈ 7 milyon | Inception modülü, 1×1 darboğaz, global ortalama havuzlama |
| ResNet-152 (2015) | %3.6 (topluluk) | 152 | ≈ 60 milyon | artık (residual) bağlantı |
| SENet (2017) | %2.25 | — | — | kanal dikkati (sıkıştır-uyar, M4.8) |

(ILSVRC: ImageNet Large Scale Visual Recognition Challenge, 1000 sınıf; "top-5 hata": doğru sınıf modelin ilk 5 tahmininde değilse hata. Yarışma sonuçları genelde birden çok modelin topluluğuyla elde edildi.)

### LeNet-5 {#lenet}

LeNet-5 (LeCun ve ark. 1998) belki de en bilinen CNN mimarisidir; el yazısı rakam tanımada (MNIST, banka çekleri) yaygın kullanıldı.

<figure class="fig wide"><svg viewBox="0 0 590 200" role="img" aria-label="LeNet-5: 32 çarpı 32 girdi, iki konvolüsyon ve iki ortalama havuzlama katmanı, 120 haritalı konvolüsyon, 84 nöronlu tam bağlı katman ve 10 çıktı"><defs><marker id="ln1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="10" y="68.0" width="52" height="64" rx="4" class="mute"/><text x="36" y="150" text-anchor="middle" class="sm">1@32×32</text><text x="36" y="166" text-anchor="middle" class="sm">girdi</text><line x1="63" y1="100" x2="79" y2="100" class="ink" marker-end="url(#ln1)"/><rect x="82" y="72.0" width="52" height="56" rx="4" class="soft-a ca"/><text x="108" y="150" text-anchor="middle" class="sm">6@28×28</text><text x="108" y="180" text-anchor="middle" class="sm">C1 conv 5×5</text><line x1="135" y1="100" x2="151" y2="100" class="ink" marker-end="url(#ln1)"/><rect x="154" y="80.0" width="52" height="40" rx="4" class="soft-b cb"/><text x="180" y="150" text-anchor="middle" class="sm">6@14×14</text><text x="180" y="166" text-anchor="middle" class="sm">S2 ort. havuz</text><line x1="207" y1="100" x2="223" y2="100" class="ink" marker-end="url(#ln1)"/><rect x="226" y="84.0" width="52" height="32" rx="4" class="soft-a ca"/><text x="252" y="150" text-anchor="middle" class="sm">16@10×10</text><text x="252" y="180" text-anchor="middle" class="sm">C3 conv 5×5</text><line x1="279" y1="100" x2="295" y2="100" class="ink" marker-end="url(#ln1)"/><rect x="298" y="89.0" width="52" height="22" rx="4" class="soft-b cb"/><text x="324" y="150" text-anchor="middle" class="sm">16@5×5</text><text x="324" y="166" text-anchor="middle" class="sm">S4 ort. havuz</text><line x1="351" y1="100" x2="367" y2="100" class="ink" marker-end="url(#ln1)"/><rect x="370" y="93.0" width="52" height="14" rx="4" class="soft-a ca"/><text x="396" y="150" text-anchor="middle" class="sm">120@1×1</text><text x="396" y="180" text-anchor="middle" class="sm">C5 conv 5×5</text><line x1="423" y1="100" x2="439" y2="100" class="ink" marker-end="url(#ln1)"/><rect x="442" y="93.0" width="52" height="14" rx="4" class="soft-p cp"/><text x="468" y="150" text-anchor="middle" class="sm">84</text><text x="468" y="166" text-anchor="middle" class="sm">F6 tam bağlı</text><line x1="495" y1="100" x2="511" y2="100" class="ink" marker-end="url(#ln1)"/><rect x="514" y="93.0" width="52" height="14" rx="4" class="soft-p cp"/><text x="540" y="150" text-anchor="middle" class="sm">10</text><text x="540" y="180" text-anchor="middle" class="sm">çıktı (RBF)</text></svg><figcaption>Şekil 1 — LeNet-5 (LeCun ve ark. 1998). 28×28 MNIST rakamları 32×32'ye sıfır dolgulanır; sonraki katmanlar dolgu kullanmadığı için boyut her konvolüsyonda 4 piksel küçülür (5×5 çekirdek), her havuzlamada yarıya iner. Aktivasyonlar tanh (orijinalde ölçeklenmiş tanh); havuzlamadan sonra da aktivasyon uygulanır.</figcaption></figure>

- MNIST görüntüleri 28×28'dir ama ağa verilmeden önce **32×32'ye sıfır dolgulanır** ve normalize edilir. Ağın geri kalanı dolgu kullanmaz; bu yüzden boyut ağ boyunca küçülür: $32-5+1=28$, havuzla 14, $14-5+1=10$, havuzla 5, $5-5+1=1$.
- **Ortalama havuzlama** katmanları bugünkülerden biraz daha karmaşıktır: her nöron girdilerinin ortalamasını hesaplar, sonucu **öğrenilen** bir katsayıyla (harita başına bir) çarpar, öğrenilen bir bias (yine harita başına) ekler ve aktivasyon uygular.
- **C3'teki** nöronların çoğu S2'deki altı haritanın hepsine değil yalnız üç ya da dördüne bağlıdır (o dönemin hesap kısıtı ve simetriyi kırma gerekçesiyle; bugün tam bağlantı kullanılır).
- **Çıktı katmanı** biraz özeldir: girdi vektörü ile ağırlık vektörünün iç çarpımı yerine, her nöron ikisi arasındaki **Öklid uzaklığının karesini** verir (RBF — radyal taban fonksiyonu birimi). Her çıktı, görüntünün belirli bir rakam sınıfına ne kadar ait olduğunu ölçer; RBF aslında bir aktivasyon fonksiyonu değil, bir olasılık yoğunluğu kestiricisi gibidir. Bugün **çapraz entropi** tercih edilir: kötü tahminleri çok daha sert cezalandırır, daha büyük gradyanlar üretir ve daha hızlı yakınsar (M3.2).

### AlexNet ve veri artırma {#alexnet}

AlexNet (Krizhevsky, Sutskever ve Hinton 2012) ILSVRC-2012'yi büyük farkla kazandı: top-5 hata oranı **%15.3**, ikinci en iyi **%26.2** (makalede ILSVRC-2010 test kümesinde %17.0'ye karşı %25.7 de raporlanır; ders slaytlarındaki "%17'ye karşı %26" bu iki sonucun karışımıdır). LeNet-5'e benzer ama çok daha büyük ve derindir: 5 konvolüsyon ve 3 tam bağlı katman, ≈ 60 milyon parametre, 650 000 nöron. Konvolüsyon katmanlarını her birinin üstüne bir havuzlama koymak yerine **doğrudan üst üste** yığan ilk ağdır. Başarısının bileşenleri:

- **ReLU** (sigmoid/tanh yerine): doymayan aktivasyon, birkaç kat daha hızlı eğitim (M3.5).
- **İki GPU** üzerinde eğitim (dönemin 3 GB'lık kartlarına sığmak için model ikiye bölündü).
- Aşırı uyumu azaltmak için iki düzenlileştirme: (1) son iki tam bağlı katmanın (F9, F10) çıktılarına eğitimde **%50 dropout** (M3.7); (2) **veri artırma**: eğitim görüntülerini rastgele ötelemek, yatay çevirmek ve ışık koşullarını değiştirmek.
- **Yerel tepki normalizasyonu** (LRN, aşağıda).

**Veri artırma** (data augmentation), her eğitim örneğinin birçok **gerçekçi** varyantını üreterek eğitim kümesini yapay olarak büyütür. Aşırı uyumu azaltır; bir düzenlileştirme tekniğidir. Üretilen örnekler olabildiğince gerçekçi olmalıdır: ideal olarak bir insan, artırılmış kümedeki bir görüntünün artırılıp artırılmadığını ayırt edememelidir. Örneğin her resmi biraz kaydır, döndür ve yeniden boyutlandır: model nesnelerin konumundaki, yönelimindeki ve boyutundaki değişimlere daha toleranslı olmaya zorlanır. Farklı ışık koşullarına toleranslı bir model için farklı kontrastlarda görüntüler üret. Resimleri yatay çevirmek de genelde işe yarar (metin ve diğer asimetrik nesneler hariç — "b" çevrilince "d" olur). Bu dönüşümleri birleştirerek eğitim kümesini büyük ölçüde çoğaltabilirsin. PyTorch'ta `torchvision.transforms.v2` (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, …); dönüşümler her epoch'ta yeniden örneklenir, dolayısıyla ağ aynı görüntüyü hiç iki kez aynı biçimde görmez.

:::ayrıntı Gürültü eklemek işe yaramaz mı?
Ders slaytları (Géron'u izleyerek) "yalnızca beyaz gürültü eklemek yardım etmez; değişiklikler öğrenilebilir olmalı" der. Kastedilen: veri artırmanın asıl gücü, test zamanında gerçekten karşılaşılacak **yapısal** çeşitliliği (konum, ölçek, ışık) öğretmesidir; bağımsız piksel gürültüsü bu çeşitliliği taklit etmez. Ama gürültü tamamen yararsız değildir: hafif Gauss gürültüsü bir düzenlileştirici gibi davranır (girdiye gürültü eklemek doğrusal modelde ridge'e denktir; Bishop 1995) ve sensör gürültüsüne karşı sağlamlık kazandırır. Etkisi genelde geometrik ve fotometrik artırmalardan zayıftır.
:::

:::ayrıntı Yerel tepki normalizasyonu (LRN)
AlexNet, C1 ve C3 katmanlarının ReLU adımından hemen sonra **rekabetçi** bir normalizasyon uygular: en güçlü etkinleşen nöronlar, **komşu öznitelik haritalarında aynı konumdaki** nöronları bastırır (biyolojik nöronlarda gözlenen yanal ketlemeye benzer). Bu, farklı haritaların uzmanlaşmasını teşvik eder, onları birbirinden uzaklaştırıp daha geniş bir öznitelik yelpazesini keşfetmeye zorlar.
$$b_i=a_i\Big(k+\alpha\sum_{j=j_{\text{low}}}^{j_{\text{high}}}a_j^2\Big)^{-\beta},\qquad j_{\text{high}}=\min\big(i+\tfrac r2,\,f_n-1\big),\quad j_{\text{low}}=\max\big(0,\,i-\tfrac r2\big).$$
$b_i$: $i$. haritadaki (belirli bir satır ve sütundaki) nöronun normalize çıktısı; $a_i$: ReLU'dan sonra, normalizasyondan önceki aktivasyon; $f_n$: harita sayısı; $k$ (bias), $\alpha$, $\beta$, $r$ (derinlik yarıçapı) hiperparametreler. Örneğin $r=2$ ise güçlü bir nöron hemen üstündeki ve altındaki haritalardaki nöronları bastırır. **Hiperparametreler:** AlexNet makalesinde $k=2$, $n=5$ komşu harita (her yönde 2), $\alpha=10^{-4}$, $\beta=0.75$. (Ders slaytındaki "$\alpha=0.0002$, $k=1$" değerleri, TensorFlow'un `local_response_normalization` tanımına göre farklı ölçeklenmiş bir aktarımdan gelir ve makaleyle uyuşmaz; PyTorch'un `nn.LocalResponseNorm(size=5, alpha=5e-4, beta=0.75, k=2)` çağrısı α'yı 5'e böldüğü için makaledekiyle aynıdır, M4.8.) VGG makalesi LRN'nin performansı artırmadığını bildirdi; BatchNorm'un (M3.6) gelişiyle LRN kullanımdan kalktı.
:::

### VGG: yalnız 3×3 {#vgg}

ILSVRC-2014'ün ikincisi VGGNet'tir (Simonyan ve Zisserman, Oxford Visual Geometry Group). Çok basit ve klasik bir mimarisi vardır: 2 ya da 3 konvolüsyon katmanı ve bir havuzlama, yine 2 ya da 3 konvolüsyon ve bir havuzlama… (toplam 16 ya da 19 ağırlıklı katman: VGG-11, -13, -16, -19 yapılandırmaları), sonda 2 gizli katmanlı yoğun bir ağ ve çıktı katmanı. Tek boyutta çok sayıda filtre kullanır: **3×3**.

<figure class="fig wide"><svg viewBox="0 0 560 205" role="img" aria-label="İki üç çarpı üçlük katmanın üst üste alıcı alanı beş pikseldir"><rect x="30" y="150" width="22" height="22" rx="3" class="mute"/><rect x="56" y="150" width="22" height="22" rx="3" class="soft-a ca"/><rect x="82" y="150" width="22" height="22" rx="3" class="soft-a ca"/><rect x="108" y="150" width="22" height="22" rx="3" class="soft-a ca"/><rect x="134" y="150" width="22" height="22" rx="3" class="soft-a ca"/><rect x="160" y="150" width="22" height="22" rx="3" class="soft-a ca"/><rect x="186" y="150" width="22" height="22" rx="3" class="mute"/><text x="224" y="166" class="sm">girdi</text><rect x="56" y="95" width="22" height="22" rx="3" class="mute"/><rect x="82" y="95" width="22" height="22" rx="3" class="soft-p cp"/><rect x="108" y="95" width="22" height="22" rx="3" class="soft-p cp"/><rect x="134" y="95" width="22" height="22" rx="3" class="soft-p cp"/><rect x="160" y="95" width="22" height="22" rx="3" class="mute"/><text x="224" y="111" class="sm">1. katman (3×3)</text><rect x="82" y="40" width="22" height="22" rx="3" class="mute"/><rect x="108" y="40" width="22" height="22" rx="3" class="soft-p cp"/><rect x="134" y="40" width="22" height="22" rx="3" class="mute"/><text x="224" y="56" class="sm">2. katman (3×3)</text><line x1="93" y1="117" x2="67" y2="150" class="mute"/><line x1="93" y1="117" x2="93" y2="150" class="mute"/><line x1="93" y1="117" x2="119" y2="150" class="mute"/><line x1="119" y1="117" x2="93" y2="150" class="mute"/><line x1="119" y1="117" x2="119" y2="150" class="mute"/><line x1="119" y1="117" x2="145" y2="150" class="mute"/><line x1="145" y1="117" x2="119" y2="150" class="mute"/><line x1="145" y1="117" x2="145" y2="150" class="mute"/><line x1="145" y1="117" x2="171" y2="150" class="mute"/><line x1="119" y1="62" x2="93" y2="95" class="cp"/><line x1="119" y1="62" x2="119" y2="95" class="cp"/><line x1="119" y1="62" x2="145" y2="95" class="cp"/><text x="30" y="196" class="sm">2. katmanın bir nöronu girdinin 5 pikselini görür: iki 3×3 = bir 5×5 alıcı alan (2-B'de 25 piksel)</text></svg><figcaption>Şekil 3 — VGG fikri, 1-B kesitte. İki 3×3 katman 5×5, üç 3×3 katman 7×7 alıcı alan verir. C kanallı katmanlarda parametre: iki 3×3 için 18C², bir 5×5 için 25C²; üç 3×3 için 27C², bir 7×7 için 49C² — üstelik araya giren her ReLU fonksiyonu daha doğrusalsız yapar.</figcaption></figure>

:::türetim Neden küçük çekirdekler?
Adım 1'de $n$ tane 3×3 katmanın alıcı alanı $1+2n$'dir (her katman her yöne 1 piksel ekler): iki katman 5×5, üç katman 7×7. Girdi ve çıktı kanalı $C$ olan katmanlarda (bias'ı yok say):
$$\underbrace{n\cdot9C^2}_{n\text{ tane }3\times3}\quad\text{vs}\quad\underbrace{(2n+1)^2C^2}_{\text{tek büyük çekirdek}}:\qquad 18C^2<25C^2,\quad 27C^2<49C^2 .$$
Hesap da aynı oranda azalır. Üstelik aradaki her ReLU, karar fonksiyonunu daha ayırt edici yapar: tek büyük çekirdek alıcı alanda tek bir doğrusal sınıflandırıcı gezdirir, yığın küçük bir derin ağ gezdirir. Bedeli: daha çok katman, daha çok ara aktivasyon belleği.
:::

VGG'nin parametrelerinin büyük kısmı ilk tam bağlı katmandadır: son havuzlamadan çıkan $7\times7\times512$ değer 4096 nörona bağlanır, $7\cdot7\cdot512\cdot4096\approx102.8$ milyon ağırlık — VGG-16'nın 138 milyon parametresinin dörtte üçü. GoogLeNet ve ResNet'in global ortalama havuzlaması bu katmanı ortadan kaldırır.

### Alıcı alanı hesaplamak {#alici}

Bir nöronun alıcı alanı, girdi görüntüsünde değerini etkileyebilen bölgedir. Katman katman hesaplanır:

:::türetim Alıcı alan yinelemesi
$l$. katmandan sonra iki büyüklüğü izle: alıcı alan $r_l$ (piksel) ve **toplam adım** $j_l$ (bu katmandaki komşu iki nöron arasındaki uzaklığın girdi pikseli cinsinden karşılığı). Başlangıç: $r_0=1$, $j_0=1$. Çekirdeği $k_l$, adımı $s_l$ olan bir katman için:
$$r_l=r_{l-1}+(k_l-1)\,j_{l-1},\qquad j_l=j_{l-1}\,s_l .$$
**Neden:** yeni nöron önceki katmanın $k_l$ komşu nöronunu görür; bunlar girdi uzayında $j_{l-1}$ aralıklıdır, dolayısıyla uç nöronların merkezleri arasındaki uzaklık $(k_l-1)j_{l-1}$; her birinin kendi alanı $r_{l-1}$ olduğundan toplam $r_{l-1}+(k_l-1)j_{l-1}$. Genişleme $d$ ile $k_l\to d(k_l-1)+1$. Havuzlama da bir katmandır (2×2, adım 2: $k=2$, $s=2$).
:::

**Örnek:** VGG-16'nın son konvolüsyonu (conv5_3; 13 konvolüsyon, araya 4 havuzlama) 196×196 piksellik bir alanı görür; son havuzlamadan sonra 212. Hesabın ilginç sonucu: havuzlamadan **sonra** gelen her 3×3 katman alanı daha çok büyütür, çünkü $j$ her havuzlamada ikiye katlanır.

Teorik alan her pikselin **eşit** katkı verdiği anlamına gelmez. Merkezdeki pikselden çıktıya çok daha fazla yol vardır; art arda konvolüsyonların birleşimi (merkezi limit teoremi gibi) **Gauss benzeri** bir katkı profili üretir. Luo ve ark. (2016) bu **etkin alıcı alanın** teorik alanın yalnız bir kısmını kapladığını ve eğitimle büyüdüğünü gösterdi. Aşağıdaki deneyde bir mimari kur: teorik alanı, harita boyunu ve son katmanın merkez nöronunun girdiye göre gradyanını (1-B kesit) gör.

::lab l1

### GoogLeNet: Inception ve 1×1 konvolüsyonlar {#googlenet}

GoogLeNet (Szegedy ve ark., Google) ILSVRC-2014'ü top-5 hatayı **%7'nin altına** (%6.7) indirerek kazandı. Bu performansın büyük kısmı ağın önceki CNN'lerden **çok daha derin** (22 katman) olmasından geliyordu; bunu mümkün kılan, parametreleri çok daha verimli kullanan **Inception modülleriydi**: GoogLeNet, AlexNet'ten **≈ 10 kat daha az** parametreye sahiptir (≈ 6–7 milyon, 60 milyon yerine).

Bir Inception modülünde girdi dört farklı kola kopyalanır: 1×1 konvolüsyon; 1×1 sonra 3×3; 1×1 sonra 5×5; 3×3 max havuzlama sonra 1×1. Farklı çekirdek boyları **farklı ölçeklerdeki** desenleri yakalar. Tüm katmanlar adım 1 ve "same" dolgu kullandığı için (havuzlama dahil) çıktıların yüksekliği ve genişliği aynıdır ve **derinlik boyunca birleştirilebilir** (`torch.cat(..., dim=1)`). Bütün modülü, çeşitli ölçeklerde karmaşık desenleri yakalayan "steroidli bir konvolüsyon katmanı" gibi düşünebilirsin.

**1×1 konvolüsyonların üç işlevi:** (1) uzamsal desen yakalayamazlar ama **derinlik (kanal) boyunca** desen yakalarlar; (2) girdilerinden daha az harita üretecek biçimde yapılandırılırlar: **darboğaz** katmanlarıdır, boyutu düşürür, hesabı ve parametreyi azaltır, eğitimi hızlandırır ve genellemeyi iyileştirir; (3) her [1×1, 3×3] ve [1×1, 5×5] çifti tek, daha güçlü bir konvolüsyon katmanı gibi davranır: görüntü üzerinde basit bir doğrusal sınıflandırıcı yerine **iki katmanlı bir sinir ağı** gezdirir. Her Inception modülünde 6 çekirdek sayısı hiperparametresi vardır. (Modülün ayrıntıları, FLOP hesabı ve sonraki sürümleri M4.8'de.)

**Mimarinin akışı:** ilk iki katman görüntünün yüksekliğini ve genişliğini 4'e böler (alan 16'ya iner) — hesap yükünü azaltmak için; ilk katman çok bilgi korunsun diye büyük bir çekirdek (7×7, adım 2) kullanır, ardından LRN. Sonra biri darboğaz olan iki konvolüsyon (tek bir akıllı konvolüsyon gibi), yine LRN, bir max havuzlama. Ardından araya birkaç max havuzlama serpiştirilmiş **dokuz Inception modülü**. Sonra **global ortalama havuzlama** her haritanın ortalamasını verir: kalan uzamsal bilgiyi atar — sorun değil, çünkü 224×224'lük girdi beş havuzlamadan sonra 7×7'ye inmiştir ve görev yerelleştirme değil sınıflandırmadır. Bu boyut indirgemesi sayesinde AlexNet'teki gibi birkaç büyük tam bağlı katmana gerek kalmaz: parametre sayısı ve aşırı uyum riski büyük ölçüde azalır. Sonda düzenlileştirme için dropout, 1000 birimli bir tam bağlı katman ve softmax. Sonraki sürümler (Inception-v3, -v4) biraz farklı modüllerle daha iyi sonuçlara ulaştı.

### ResNet: artık öğrenme {#resnet}

Kaiming He ve ark. ILSVRC-2015'i **%3.6'nın altında** top-5 hatayla (%3.57) bir **Artık Ağla** (Residual Network, ResNet) kazandı. Kazanan sürüm 152 katmanlıydı (diğerleri 18, 34, 50, 101). Genel eğilimi doğruladı: modeller giderek daha derin, parametreleri giderek daha az.

**Bozulma problemi.** Neden yalnız katman ekleyerek daha derine gidilemiyordu? He ve ark. düz (artıksız) ağlarda şunu gözlemledi: 56 katmanlı ağın **eğitim** hatası 20 katmanlıdan daha yüksekti. Bu aşırı uyum değildir (eğitim hatası daha yüksek!); bir **optimizasyon** sorunudur. Oysa derin ağ en azından sığ ağ kadar iyi olabilmeliydi: fazladan katmanların kimlik fonksiyonunu öğrenmesi yeterdi. Demek ki birkaç doğrusalsız katmanın kimliği öğrenmesi zordur.

**Artık öğrenme.** Bir ağın amacı bir hedef fonksiyon $h(x)$'i modellemekse, girdi $x$'i ağın çıktısına eklersen (bir **atlama** ya da **kısayol** bağlantısı), ağ $h(x)$ yerine $f(x)=h(x)-x$'i modellemeye zorlanır. Sıradan bir ağı başlattığında ağırlıklar sıfıra yakındır ve ağ sıfıra yakın değerler verir; atlama bağlantıları eklersen ağ başlangıçta girdisinin bir kopyasını verir, yani **kimlik fonksiyonunu** modeller. Hedef fonksiyon kimliğe oldukça yakınsa (çoğu zaman öyledir) bu eğitimi önemli ölçüde hızlandırır. Ayrıca çok sayıda atlama bağlantısı varsa ağ, birkaç katman henüz öğrenmeye başlamamışken bile ilerleme kaydedebilir: sinyal tüm ağı kolayca geçer.

<figure class="fig"><svg viewBox="0 0 420 268" role="img" aria-label="Artık blok: iki üç çarpı üç konvolüsyon ve BatchNorm katmanı, girdiyi çıktıya ekleyen kimlik kısayolu, ardından ReLU"><defs><marker id="rb1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="rb2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cq-f"/></marker></defs><text x="190" y="16" text-anchor="middle">x (C kanal)</text><rect x="120" y="30" width="140" height="26" rx="5" class="soft-a ca"/><text x="190" y="47" text-anchor="middle" class="sm">conv 3×3, C, adım 1</text><rect x="120" y="66" width="140" height="26" rx="5" class="soft-p cp"/><text x="190" y="83" text-anchor="middle" class="sm">BN + ReLU</text><rect x="120" y="102" width="140" height="26" rx="5" class="soft-a ca"/><text x="190" y="119" text-anchor="middle" class="sm">conv 3×3, C, adım 1</text><rect x="120" y="138" width="140" height="26" rx="5" class="soft-p cp"/><text x="190" y="155" text-anchor="middle" class="sm">BN</text><line x1="190" y1="56" x2="190" y2="65" class="ink" marker-end="url(#rb1)"/><line x1="190" y1="92" x2="190" y2="101" class="ink" marker-end="url(#rb1)"/><line x1="190" y1="128" x2="190" y2="137" class="ink" marker-end="url(#rb1)"/><line x1="190" y1="20" x2="190" y2="29" class="ink" marker-end="url(#rb1)"/><circle cx="190" cy="186" r="11" class="ink"/><text x="190" y="191" text-anchor="middle">+</text><line x1="190" y1="164" x2="190" y2="174" class="ink" marker-end="url(#rb1)"/><path d="M 190 22 H 320 V 186 H 202" class="cq dash" fill="none" marker-end="url(#rb2)"/><text x="326" y="110" class="sm tq">kimlik (atlama)</text><rect x="120" y="210" width="140" height="26" rx="5" class="soft-p cp"/><text x="190" y="227" text-anchor="middle" class="sm">ReLU</text><line x1="190" y1="197" x2="190" y2="209" class="ink" marker-end="url(#rb1)"/><text x="190" y="256" text-anchor="middle">h(x) = ReLU(F(x) + x)</text><text x="10" y="100" class="sm">F(x): öğrenilen</text><text x="10" y="114" class="sm">artık (residual)</text><text x="10" y="232" class="sm">boyut değişirse:</text><text x="10" y="246" class="sm">kısayol = 1×1 conv,</text><text x="10" y="260" class="sm">adım 2 (+ BN)</text></svg><figcaption>Şekil 2 — ResNet'in temel artık bloğu (BasicBlock, ResNet-18/34). Ağ h(x) yerine F(x) = h(x) − x'i öğrenir. Ağırlıklar sıfıra yakınken blok kimliğe yakındır; kimlik yolu hem sinyali hem gradyanı hiçbir ağırlık çarpanından geçirmeden taşır. Kanal sayısı ikiye katlanıp boyut yarıya inerken kısayol 1×1, adım 2 konvolüsyonla izdüşürülür.</figcaption></figure>

:::türetim Gradyan neden sönmez? (ders slaytının sorusu)
**Soru:** Gradyan sorunlarına hangi çarpanlar neden oluyordu? Artık bağlantılarda bu çarpanlar var mı?

Düz ağda $h_{l+1}=\phi(W_lh_l)$ ve zincir kuralı $\frac{\partial L}{\partial h_l}=\frac{\partial L}{\partial h_L}\prod_{k=l}^{L-1}\frac{\partial h_{k+1}}{\partial h_k}$ verir: her çarpan $W_k^\top\mathrm{diag}(\phi')$ biçimindedir. Çarpanların tipik büyüklüğü 1'den küçükse gradyan **üstel olarak söner**, büyükse **patlar** (M3.2, M3.5).

Artık blokta $h_{l+1}=h_l+F(h_l)$, dolayısıyla $\frac{\partial h_{l+1}}{\partial h_l}=I+\frac{\partial F}{\partial h_l}$. Blok sonrası aktivasyon olmayan ("tam kimlik" kısayollu, He ve ark. 2016b) bir ağda açınca:
$$h_L=h_l+\sum_{k=l}^{L-1}F(h_k)\quad\Rightarrow\quad\frac{\partial L}{\partial h_l}=\frac{\partial L}{\partial h_L}\Big(I+\frac{\partial}{\partial h_l}\sum_{k=l}^{L-1}F(h_k)\Big).$$
Birinci terim $\frac{\partial L}{\partial h_L}$ **hiçbir ağırlık çarpanından geçmeden** her katmana ulaşır; ikinci terim bir mini-batch'te tam olarak $-I$'yı iptal etmedikçe gradyan sıfırlanamaz. Cevap: sönmeye yol açan $W^\top\mathrm{diag}(\phi')$ çarpımları kimlik yolunda yoktur. (Ama her blok varyans ekler: ileri geçişte $\mathrm{Var}(h_L)\approx\mathrm{Var}(h_l)+\sum\mathrm{Var}(F)$; bu yüzden BatchNorm ya da dalları küçük ölçekle başlatma — Fixup, M3.6 — gerekir.)
:::

Aşağıdaki deneyde aynı ağırlıklarla düz ve artık ağın katman katman gradyan normunu ve aktivasyon büyüklüğünü karşılaştır; başlatma çarpanını 1'den uzaklaştır ve artık dalın ölçeğini (α) değiştir.

::lab l2

**ResNet'in mimarisi** şaşırtıcı derecede basittir: GoogLeNet gibi başlar ve biter (dropout'suz), arada çok derin bir **artık birim** (residual unit) yığını vardır. Her birim iki konvolüsyon katmanından oluşur (havuzlama yok!): BatchNorm ve ReLU ile, 3×3 çekirdekler, uzamsal boyutları korur (adım 1, "same" dolgu). Birkaç birimde bir harita sayısı **iki katına** çıkarken yükseklik ve genişlik **yarıya** iner (adım 2'li bir konvolüsyonla). Bu olduğunda girdiler artık birimin çıktısına doğrudan eklenemez — şekilleri farklıdır. Çözüm: girdiler, adımı 2 ve çıktı harita sayısı doğru olan bir **1×1 konvolüsyondan** geçirilir (izdüşüm kısayolu).

| ağ | birim türü | aşamalardaki birim sayısı | harita sayıları |
|---|---|---|---|
| ResNet-34 | iki 3×3 (BasicBlock) | 3, 4, 6, 3 | 64, 128, 256, 512 |
| ResNet-50 | darboğaz (1×1 → 3×3 → 1×1) | 3, 4, 6, 3 | 256, 512, 1024, 2048 |
| ResNet-152 | darboğaz | 3, 8, 36, 3 | 256, 512, 1024, 2048 |

(ResNet-34 adını yalnız konvolüsyon katmanlarını ve tam bağlı katmanı sayarak alır: $1+2\cdot(3+4+6+3)+1=34$.) ResNet-50 ve daha derinleri biraz farklı birimler kullanır: 256 haritalı iki 3×3 katman yerine üç katman — önce yalnız 64 haritalı bir 1×1 (4 kat az; darboğaz), sonra 64 haritalı bir 3×3, sonra özgün derinliği geri getiren 256 haritalı bir 1×1. ResNet-152'de 256 harita çıkaran 3 birim, 512 harita çıkaran 8, tam **36** birim 1 024 harita ve 2 048 harita çıkaran 3 birim vardır. Google'ın Inception-v4'ü GoogLeNet ile ResNet fikirlerini birleştirip ImageNet'te ≈ %3 top-5 hataya ulaştı.

**Çıkarımda BatchNorm'u katlamak.** ResNet'te her konvolüsyonu bir BN izler. Çıkarımda BN sabit bir afin dönüşümdür ve önceki konvolüsyona çıktı kanalı başına katlanabilir: $W'_f=W_f\,\gamma_f/\sqrt{\sigma_f^2+\varepsilon}$, $b'_f=\gamma_f(b_f-\mu_f)/\sqrt{\sigma_f^2+\varepsilon}+\beta_f$ (türetim M3.6). Bu modülün son aşamasında bunu kodluyorsun.

### Sonrası: Xception, SENet ve ötesi {#sonrasi}

- **Xception** (Chollet 2016): Inception modüllerini **derinlemesine ayrılabilir konvolüsyonla** (her kanala ayrı bir uzamsal filtre, ardından kanalları karıştıran 1×1) değiştirir; uzamsal ve kanallar arası desenlerin ayrı modellenebileceği varsayımına dayanır. 350 milyon görüntülük, 17 000 sınıflı bir görevde Inception-v3'ü belirgin biçimde geçti.
- **SENet** (Hu ve ark. 2017): her artık birime ya da Inception modülüne küçük bir **sıkıştır-uyar** bloğu ekleyerek hangi öznitelik haritalarının birlikte etkin olduğunu öğrenir ve haritaları yeniden ölçekler; ILSVRC-2017'yi %2.25 top-5 hatayla kazandı.
- Bu blokların matematiği, parametre/FLOP hesapları ve MobileNet, ResNeXt, EfficientNet gibi verimli mimariler M4.8'de; Vision Transformer M4.15'te.

### Özet {#ozet}

- LeNet-5: konvolüsyon + havuzlama + tam bağlı; dolgusuz katmanlarla küçülen haritalar, öğrenilen ortalama havuzlama, RBF çıktı (bugün CE).
- AlexNet: büyük ölçek + GPU + ReLU + dropout + veri artırma; ardışık konvolüsyonlar; LRN (bugün yerini BN aldı).
- VGG: yalnız 3×3; $n$ katmanın alanı $1+2n$, parametre $9nC^2$ vs $(2n+1)^2C^2$; parametrelerin çoğu ilk FC katmanında.
- Alıcı alan: $r\leftarrow r+(k-1)j$, $j\leftarrow js$; etkin alan Gauss benzeri ve teorik alandan küçük.
- GoogLeNet: Inception (çok ölçek, 1×1 darboğaz), global ortalama havuzlama, AlexNet'ten 10 kat az parametre.
- ResNet: düz ağlardaki bozulma bir optimizasyon sorunudur; $h=x+F(x)$ ile Jacobian $I+\partial F$ olur, gradyanın kimlik yolu hiç çarpan görmez; boyut değişince 1×1 adım 2 izdüşüm; derin sürümlerde darboğaz birimi.
