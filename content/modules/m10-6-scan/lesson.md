### Tarama {#tarama}

**Tarama** (scan), **önek toplamı** adıyla da bilinir. Kaynak ayırma, iş atama ve polinom değerlendirme gibi ilk bakışta sıralı görünen işlemleri paralelleştirmek için sık kullanılan genel bir işlemdir. Bir hesap doğal olarak her öğenin bir öncekine bağlı tanımlandığı bir özyinelemeyle anlatılıyorsa, büyük olasılıkla paralel bir taramayla yapılabilir. Tarama, bazı paralel algoritmaların sıralı algoritmadan **daha fazla iş** yaptığı tipik bir örnektir: algoritma karmaşıklığı ile paralelleştirme arasında dikkatle kurulması gereken bir ödünleşim vardır.

İndirgemede olduğu gibi bir işleç $\oplus$ gerekir; birleşmeli olmalı ve bir birim elemanı $I$ bulunmalıdır. Taramanın iki türü vardır:

- **Kapsayıcı tarama** $[x_0,x_1,\dots,x_{n-1}]$ girdisinden $[x_0,\;x_0\oplus x_1,\;\dots,\;x_0\oplus\dots\oplus x_{n-1}]$ üretir: her çıktı karşılık gelen girdinin etkisini **içerir**. İşleç $+$ ise çıktı integral dizidir (M10.3).
- **Dışlayıcı tarama** $[I,\;x_0,\;x_0\oplus x_1,\;\dots,\;x_0\oplus\dots\oplus x_{n-2}]$ üretir: her çıktı karşılık gelen girdiyi **dışlar**.

İkisi arasında dönüşüm kolaydır. Kapsayıcıdan dışlayıcıya bütün öğeler sağa kaydırılır ve 0. konuma birim eleman yazılır. Dışlayıcıdan kapsayıcıya sola kaydırılır ve son öğe eklenir. $x=[3,1,7,0,4,1,6,3]$ için kapsayıcı tarama $[3,4,11,11,15,16,22,25]$, dışlayıcı $[0,3,4,11,11,15,16,22]$'dir.

**Sıralı tarama** `y[0] = x[0]; for (i = 1; i < n; i++) y[i] = y[i-1] ⊕ x[i];` ile yapılır. İşi girdi boyuyla doğrusal orantılıdır, $O(n)$; $n-1$ işlem yapar. Tarama dinamik programlamayla ilişkilidir: her çıktı bir öncekinin saklanmış sonucunu kullanır.

### Naif paralel tarama {#naif}

İlk akla gelen paralel yol, her çıktı öğesi için ayrı bir indirgeme yapmaktır: thread $i$, $x_0\oplus\dots\oplus x_i$'yi kendisi hesaplar. Bütün çıktılar paralel hesaplanır, ama bu yol sıralı taramadan hızlı değildir. $n$. çıktının hesabı $n$ adım sürer, sıralı kodla aynı; paralel programın süresi en uzun thread'le sınırlıdır. Sınırlı kaynakla naif yol sıralıdan çok daha yavaş bile olabilir. Toplam iş de çok büyüktür:
$$\sum_{i=0}^{n-1}i=\frac{n(n-1)}2=O(n^2).$$
Daha çok iş yapılır ama hızlanma kazanılmaz. Sekiz öğede 28 toplama yapılır (sıralı tarama 7 yapar).

### Kogge–Stone {#ks}

Verimli bir paralel tarama için farklı çıktıların indirgeme ağaçları arasında **kısmi toplamları paylaşmak** gerekir. Kogge ve Stone'un 1973'te hızlı toplayıcı devreleri için önerdiği algoritma bunu yapar ve bugün de yüksek hızlı bilgisayar aritmetiği donanımında kullanılır.

Algoritma yerinde (in-place) bir taramadır. Başta `XY[i]` $x_i$'yi içerir. Her yinelemede adım iki katına çıkar ve her konum adım kadar gerisindekini kendine ekler: `XY[i] += XY[i - stride]` (`i ≥ stride` için). $k$ yinelemeden sonra `XY[i]`, $i$ ve öncesindeki en çok $2^k$ girdinin toplamını içerir. Bir yinelemeden sonra $x_{i-1}+x_i$, iki yinelemeden sonra $x_{i-3}+\dots+x_i$. Dizinin başında negatif indeksler sıfıra kırpılır; baştaki konumlar daha az öğe biriktirir.

CUDA çekirdeği girdiyi bölümlere (section) ayırır. Her bölüm tek bir bloğun işleyebileceği kadar küçüktür ve paylaşımlı bellekteki `XY` dizisine yüklenir. Blok boyu bölüm boyuna (`SECTION_SIZE`) eşittir ve her thread bir `XY` konumunu evriltir. Döngü konumun indirgeme ağacını dolaşır. Adım thread'in indeksinden büyük olunca o konum gereken bütün girdileri biriktirmiştir ve thread etkin olmaktan çıkar. İndirgemedeki gibi, her yinelemenin başında bir `__syncthreads()` bütün thread'lerin önceki yinelemeyi bitirmesini bekler.

:::türetim Yazma-sonrası-okuma tehlikesi
İndirgemeden önemli bir fark vardır. Her etkin thread kendi konumuyla (`XY[i]`) **başka bir thread'in konumunu** (`XY[i - stride]`) toplar. Thread $i$ sonucunu, thread $i+\text{stride}$ eski değeri okumadan önce yazarsa yeni değer öteki thread'in toplamını bozar.

Örnek, ikinci yinelemede (adım 2) thread 4 ve thread 6'dır. Thread 6, `XY[4]`'ün eski değerini ($x_3+x_4$) `XY[6]`'nın eski değerine ($x_5+x_6$) ekleyip $x_3+\dots+x_6$ üretmelidir. Thread 4 bu yinelemedeki sonucunu ($x_1+\dots+x_4$) erken yazarsa, thread 6 yeni değeri okuyup $x_1+\dots+x_6$ yazar. Üçüncü yinelemede thread 6 $x_1+x_2$'yi bir kez daha ekleyecektir ve sonuç $2x_1+2x_2+x_3+\dots+x_6$ olur. Thread 6 eski değeri thread 4 yazmadan okursa sonuç doğrudur. Sonuç zamanlamaya bağlıdır ve çalıştırmadan çalıştırmaya değişebilir: bir **yarış koşulu**.

Çözüm iki parçalıdır:

1. Her etkin thread toplamını önce özel bir **geçici değişkene** (yazmaç) yazar; böylece hiçbir eski değer üzerine yazılmaz.
2. **İkinci bir `__syncthreads()`**, bütün thread'ler eski değerleri okumadan hiçbirinin yazmaya geçmemesini sağlar; ardından geçici değer `XY`'ye yazılır.

İkinci bariyerden kaçınmanın yolu **çift tampondur**: girdi ve çıktı için ayrı diziler. Okunan ve yazılan konum farklı olduğu için tehlike ortadan kalkar; bedeli iki kat paylaşımlı bellektir.

1. Global bellekten birinci tampona yüklenir.
2. Birinci yinelemede birinciden okunup ikinciye yazılır.
3. İkinci yinelemede ikinciden okunup birinciye yazılır, ve tamponlar dönüşümlü kullanılmaya devam eder.

Çift tamponlama video işlemede de çok yaygındır.
:::

:::türetim İş verimliliği
Bir algoritmanın **iş verimliliği**, yaptığı işin hesap için gereken en az işe ne kadar yakın olduğudur. Tarama için en az iş sıralı algoritmanın $n-1$ toplamasıdır.

Kogge–Stone'da bütün thread'ler $\log_2n$ adıma kadar döner ve her yinelemede etkin olmayan thread sayısı adıma eşittir:
$$\sum_{s=1,2,4,\dots,n/2}(n-s)=n\log_2n-\sum_s s=n\log_2n-(n-1).$$

İyi haber, $O(n\log n)$'nin naif yolun $O(n^2)$'sinden çok iyi olmasıdır. Kötü haber, sıralı algoritma kadar iş verimli olmamasıdır. 512 öğelik bir bölümde Kogge–Stone 4097 toplama yapar, sıralı tarama 511: **sekiz kat** fazla iş; oran $n$ ile büyür. Buna karşılık adım sayısı $\log_2n$'dir. Sınırsız kaynakla adım sayısındaki azalma $n/\log_2n$'dir: 512 için $512/9=56.9$.
:::

### Brent–Kung {#bk}

Kogge–Stone kavramsal olarak basittir, ama bazı pratik uygulamalar için iş verimliliği düşüktür. Ara sonuçların daha fazla paylaşılması mümkündür. Bunun için ara sonuçları stratejik olarak hesaplayıp farklı thread'lere dağıtmak gerekir; bu da ek hesap adımları isteyebilir. Brent ve Kung'un (1982) toplayıcı tasarımı daha iyi iş verimli bir tarama verir. İki evresi vardır.

**İndirgeme ağacı (yukarı süpürme).** En az işlemle toplam üretilir. 16 öğede:

- birinci adımda yalnız tek indeksli öğeler güncellenir: `XY[i] += XY[i-1]`;
- ikinci adımda $4n-1$ biçimindeki indeksler (3, 7, 11, 15);
- üçüncü adımda $8n-1$ biçimindekiler (7, 15);
- dördüncü adımda yalnız `XY[15]`.

Toplam $8+4+2+1=15$ işlem; genel olarak $n/2+n/4+\dots+1=n-1$.

**Ters ağaç (aşağı süpürme).** Kısmi toplamlar onları kullanabilecek konumlara dağıtılır. Yukarı süpürmede toplamalar hep **ardışık bir aralığı** biriktirir. Her konumdaki değer bir $x_i\dots x_j$ aralığı olarak yazılabilir. Örneğin ters ağaç başlamadan `XY[11]` $x_8\dots x_{11}$'i içerir. Yukarı süpürmenin sonunda `XY[0]`, `XY[1]`, `XY[3]`, `XY[7]` ve `XY[15]` zaten tamamdır. Diğerleri eksik aralıkları almalıdır. `XY[14]` yalnız $x_{14}$'ü içerir ve `XY[7]` ($x_0\dots x_7$), `XY[11]` ($x_8\dots x_{11}$) ve `XY[13]` ($x_{12}\dots x_{13}$) değerlerini biriktirmelidir. Ters ağaç bunları büyük adımlardan küçüğe ekler: `XY[11]` `XY[7]`'yi alır, `XY[13]` `XY[11]`'i alır, `XY[14]` `XY[13]`'ü alır.

<figure class="fig wide"><svg viewBox="0 0 640 236" role="img" aria-label="Sekiz öğelik bir kapsayıcı taramada Kogge–Stone ve Brent–Kung algoritmalarının veri akışı"><text x="20" y="16" class="sm">Kogge–Stone (adım 1, 2, 4)</text><text x="30" y="34" text-anchor="middle" class="sm">3</text><text x="62" y="34" text-anchor="middle" class="sm">1</text><text x="94" y="34" text-anchor="middle" class="sm">7</text><text x="126" y="34" text-anchor="middle" class="sm">0</text><text x="158" y="34" text-anchor="middle" class="sm">4</text><text x="190" y="34" text-anchor="middle" class="sm">1</text><text x="222" y="34" text-anchor="middle" class="sm">6</text><text x="254" y="34" text-anchor="middle" class="sm">3</text><path d="M30,40 L30,70" class="mute"/><path d="M62,40 L62,70" class="mute"/><path d="M94,40 L94,70" class="mute"/><path d="M126,40 L126,70" class="mute"/><path d="M158,40 L158,70" class="mute"/><path d="M190,40 L190,70" class="mute"/><path d="M222,40 L222,70" class="mute"/><path d="M254,40 L254,70" class="mute"/><path d="M30,44 L62,66" class="ca w2"/><circle cx="62" cy="66" r="3.5" class="ca-f"/><path d="M62,44 L94,66" class="ca w2"/><circle cx="94" cy="66" r="3.5" class="ca-f"/><path d="M94,44 L126,66" class="ca w2"/><circle cx="126" cy="66" r="3.5" class="ca-f"/><path d="M126,44 L158,66" class="ca w2"/><circle cx="158" cy="66" r="3.5" class="ca-f"/><path d="M158,44 L190,66" class="ca w2"/><circle cx="190" cy="66" r="3.5" class="ca-f"/><path d="M190,44 L222,66" class="ca w2"/><circle cx="222" cy="66" r="3.5" class="ca-f"/><path d="M222,44 L254,66" class="ca w2"/><circle cx="254" cy="66" r="3.5" class="ca-f"/><text x="280" y="64" class="sm">7</text><path d="M30,70 L30,100" class="mute"/><path d="M62,70 L62,100" class="mute"/><path d="M94,70 L94,100" class="mute"/><path d="M126,70 L126,100" class="mute"/><path d="M158,70 L158,100" class="mute"/><path d="M190,70 L190,100" class="mute"/><path d="M222,70 L222,100" class="mute"/><path d="M254,70 L254,100" class="mute"/><path d="M30,74 L94,96" class="ca w2"/><circle cx="94" cy="96" r="3.5" class="ca-f"/><path d="M62,74 L126,96" class="ca w2"/><circle cx="126" cy="96" r="3.5" class="ca-f"/><path d="M94,74 L158,96" class="ca w2"/><circle cx="158" cy="96" r="3.5" class="ca-f"/><path d="M126,74 L190,96" class="ca w2"/><circle cx="190" cy="96" r="3.5" class="ca-f"/><path d="M158,74 L222,96" class="ca w2"/><circle cx="222" cy="96" r="3.5" class="ca-f"/><path d="M190,74 L254,96" class="ca w2"/><circle cx="254" cy="96" r="3.5" class="ca-f"/><text x="280" y="94" class="sm">6</text><path d="M30,100 L30,130" class="mute"/><path d="M62,100 L62,130" class="mute"/><path d="M94,100 L94,130" class="mute"/><path d="M126,100 L126,130" class="mute"/><path d="M158,100 L158,130" class="mute"/><path d="M190,100 L190,130" class="mute"/><path d="M222,100 L222,130" class="mute"/><path d="M254,100 L254,130" class="mute"/><path d="M30,104 L158,126" class="ca w2"/><circle cx="158" cy="126" r="3.5" class="ca-f"/><path d="M62,104 L190,126" class="ca w2"/><circle cx="190" cy="126" r="3.5" class="ca-f"/><path d="M94,104 L222,126" class="ca w2"/><circle cx="222" cy="126" r="3.5" class="ca-f"/><path d="M126,104 L254,126" class="ca w2"/><circle cx="254" cy="126" r="3.5" class="ca-f"/><text x="280" y="124" class="sm">4</text><text x="30" y="146" text-anchor="middle" class="sm">3</text><text x="62" y="146" text-anchor="middle" class="sm">4</text><text x="94" y="146" text-anchor="middle" class="sm">11</text><text x="126" y="146" text-anchor="middle" class="sm">11</text><text x="158" y="146" text-anchor="middle" class="sm">15</text><text x="190" y="146" text-anchor="middle" class="sm">16</text><text x="222" y="146" text-anchor="middle" class="sm">22</text><text x="254" y="146" text-anchor="middle" class="sm">25</text><text x="20" y="166" class="sm">3 adım, 17 toplama</text><text x="330" y="16" class="sm">Brent–Kung (yukarı 3, aşağı 2 adım)</text><text x="340" y="34" text-anchor="middle" class="sm">3</text><text x="372" y="34" text-anchor="middle" class="sm">1</text><text x="404" y="34" text-anchor="middle" class="sm">7</text><text x="436" y="34" text-anchor="middle" class="sm">0</text><text x="468" y="34" text-anchor="middle" class="sm">4</text><text x="500" y="34" text-anchor="middle" class="sm">1</text><text x="532" y="34" text-anchor="middle" class="sm">6</text><text x="564" y="34" text-anchor="middle" class="sm">3</text><path d="M340,40 L340,70" class="mute"/><path d="M372,40 L372,70" class="mute"/><path d="M404,40 L404,70" class="mute"/><path d="M436,40 L436,70" class="mute"/><path d="M468,40 L468,70" class="mute"/><path d="M500,40 L500,70" class="mute"/><path d="M532,40 L532,70" class="mute"/><path d="M564,40 L564,70" class="mute"/><path d="M340,44 L372,66" class="cb w2"/><circle cx="372" cy="66" r="3.5" class="cb-f"/><path d="M404,44 L436,66" class="cb w2"/><circle cx="436" cy="66" r="3.5" class="cb-f"/><path d="M468,44 L500,66" class="cb w2"/><circle cx="500" cy="66" r="3.5" class="cb-f"/><path d="M532,44 L564,66" class="cb w2"/><circle cx="564" cy="66" r="3.5" class="cb-f"/><text x="590" y="64" class="sm">4</text><path d="M340,70 L340,100" class="mute"/><path d="M372,70 L372,100" class="mute"/><path d="M404,70 L404,100" class="mute"/><path d="M436,70 L436,100" class="mute"/><path d="M468,70 L468,100" class="mute"/><path d="M500,70 L500,100" class="mute"/><path d="M532,70 L532,100" class="mute"/><path d="M564,70 L564,100" class="mute"/><path d="M372,74 L436,96" class="cb w2"/><circle cx="436" cy="96" r="3.5" class="cb-f"/><path d="M500,74 L564,96" class="cb w2"/><circle cx="564" cy="96" r="3.5" class="cb-f"/><text x="590" y="94" class="sm">2</text><path d="M340,100 L340,130" class="mute"/><path d="M372,100 L372,130" class="mute"/><path d="M404,100 L404,130" class="mute"/><path d="M436,100 L436,130" class="mute"/><path d="M468,100 L468,130" class="mute"/><path d="M500,100 L500,130" class="mute"/><path d="M532,100 L532,130" class="mute"/><path d="M564,100 L564,130" class="mute"/><path d="M436,104 L564,126" class="cb w2"/><circle cx="564" cy="126" r="3.5" class="cb-f"/><text x="590" y="124" class="sm">1</text><path d="M340,130 L340,160" class="mute"/><path d="M372,130 L372,160" class="mute"/><path d="M404,130 L404,160" class="mute"/><path d="M436,130 L436,160" class="mute"/><path d="M468,130 L468,160" class="mute"/><path d="M500,130 L500,160" class="mute"/><path d="M532,130 L532,160" class="mute"/><path d="M564,130 L564,160" class="mute"/><path d="M436,134 L500,156" class="cb w2"/><circle cx="500" cy="156" r="3.5" class="cb-f"/><text x="590" y="154" class="sm">1</text><path d="M340,160 L340,190" class="mute"/><path d="M372,160 L372,190" class="mute"/><path d="M404,160 L404,190" class="mute"/><path d="M436,160 L436,190" class="mute"/><path d="M468,160 L468,190" class="mute"/><path d="M500,160 L500,190" class="mute"/><path d="M532,160 L532,190" class="mute"/><path d="M564,160 L564,190" class="mute"/><path d="M372,164 L404,186" class="cb w2"/><circle cx="404" cy="186" r="3.5" class="cb-f"/><path d="M436,164 L468,186" class="cb w2"/><circle cx="468" cy="186" r="3.5" class="cb-f"/><path d="M500,164 L532,186" class="cb w2"/><circle cx="532" cy="186" r="3.5" class="cb-f"/><text x="590" y="184" class="sm">3</text><text x="340" y="206" text-anchor="middle" class="sm">3</text><text x="372" y="206" text-anchor="middle" class="sm">4</text><text x="404" y="206" text-anchor="middle" class="sm">11</text><text x="436" y="206" text-anchor="middle" class="sm">11</text><text x="468" y="206" text-anchor="middle" class="sm">15</text><text x="500" y="206" text-anchor="middle" class="sm">16</text><text x="532" y="206" text-anchor="middle" class="sm">22</text><text x="564" y="206" text-anchor="middle" class="sm">25</text><text x="330" y="226" class="sm">5 adım, 11 toplama</text></svg><figcaption>Şekil 1 — Aynı sonuç (3, 4, 11, 11, 15, 16, 22, 25), farklı ödünleşim: Kogge–Stone 3 adımda 17 toplama, Brent–Kung 5 adımda 11 toplama; sıralı tarama 7 adımda 7 toplama yapar. Satır sonundaki sayılar o adımdaki toplama sayısı (hesaplandı).</figcaption></figure>

**Gerçekleştirim.** Yukarı süpürmenin dolaysız yazımı (`if ((threadIdx.x + 1) % (2*stride) == 0)`) indirgemedeki basit eşleme gibi ciddi ayrışma sorunu yaşar. Ama indirgemedeki yakınsak eşleme burada kullanılamaz, çünkü taramanın ihtiyaç duyduğu **ara konumlardaki** kısmi toplamları üretmez. Çözüm daha karmaşık bir thread–veri eşlemesidir: bitişik bir thread grubu, adım kadar aralıklı konumlara eşlenir. Thread $t$ yukarı süpürmede `index = (t + 1) * 2 * stride - 1` konumunu (`index < n` ise) günceller; aşağı süpürmede `XY[index + stride] += XY[index]` ile dağıtım yapar. Etkin thread'ler hep bitişik indekslerden oluşur, ayrışma son birkaç adımla sınırlı kalır.

İki evrede de en çok $n/2$ thread gerekir. Çekirdek bölüm boyunun yarısı kadar thread'le başlatılır ve 1024 thread'lik bloklarla bölüm 2048 öğeye çıkar. Her thread başta iki öğe yükler, sonda iki öğe yazar.

**İş ve adım.** Brent–Kung toplam $2n-2-\log_2n$ toplama yapar; sıralı algoritmanın **iki katını asla aşmaz**. Adım sayısı $2\log_2n-1$'dir. Enerjisi kısıtlı bir ortamda paralellik ile verimlilik arasında iyi bir denge kurar. CUDA çekirdeğindeki üstünlüğü ise sınırlıdır. Thread'ler zaten ayrılmıştır: etkin olmayan thread'ler yürütme kaynağı tüketmez ama SM'de yer tutar. Etkin thread sayısı ağaç boyunca Kogge–Stone'dakinden çok daha hızlı düşer. Bölüm küçükken Kogge–Stone'un daha az adımı genellikle daha hızlıdır.

| algoritma | toplama | adım | n = 8 | n = 512 |
|---|---|---|---|---|
| sıralı | $n-1$ | $n-1$ | 7 / 7 | 511 / 511 |
| naif | $n(n-1)/2$ | $n-1$ | 28 / 7 | 130 816 / 511 |
| Kogge–Stone | $n\log_2n-(n-1)$ | $\log_2n$ | 17 / 3 | 4097 / 9 |
| Brent–Kung | $2n-2-\log_2n$ | $2\log_2n-1$ | 11 / 5 | 1013 / 17 |

Aşağıdaki deney üç algoritmanın adım başına toplama sayısını sayıyor. Kogge–Stone'un yerinde, tek bariyerli sürümünde rastgele zamanlamalarla yazma-sonrası-okuma yarışının sonucu nasıl bozduğunu da gösteriyor.

::lab l1

### Keyfi uzunluk: hiyerarşik tarama {#hiyerarsik}

Birçok uygulamada taranacak öğe sayısı milyonları, hatta milyarları bulur. Şimdiye kadarki çekirdekler girdinin bölümleri üzerinde yerel, blok çapında taramalar yapar; bölümlerin sonuçlarını birleştirmenin bir yolu gerekir. Bunun için **hiyerarşik tarama** kullanılır. M10.3'teki CPU blok taramasının GPU sürümüdür.

1. **Bölüm taramaları:** girdi, her biri bir SM'nin paylaşımlı belleğine sığan ve tek bir blokça işlenen bölümlere ayrılır. Her bölüm Kogge–Stone, Brent–Kung ya da kabalaştırılmış bir çekirdekle bağımsız taranır. Sonunda her öğe yalnız kendi bölümündeki öncekilerin toplamını içerir. Ders örneğinde 1. bölümün girdisi $(0,4,1,2)$, yerel taraması $(0,4,5,7)$'dir. Son sonuç için 0. bölümün toplamı ($2+1+3+1=7$) eklenmelidir.
2. **Blok toplamlarının taranması:** her bölümün son öğesi, yani bölüm toplamı, global bellekte bir $S$ dizisine yazılır. Bunu her bloğun son thread'i `S[blockIdx.x]`'e yazarak yapar, çünkü farklı blokların değerleri yalnız global bellek üzerinden görünür. $S$ tek bloklu bir taramayla taranır. Örnekte sonuç $(7,14,20,31)$'dir. Bu değerler özgün problemin "stratejik" konumlarındaki (3, 7, 11, 15) son tarama sonuçlarıdır.
3. **Ofset ekleme:** üçüncü çekirdek bölüm $b$'nin her öğesine $S[b-1]$'i ekler.

Bu üç çekirdekle yapılır. $2^{20}$ öğe ve 1024'lük bölümlerle $S$'nin uzunluğu 1024'tür. $S$ tek bir bloğa sığmazsa aynı yöntem $S$'ye özyinelemeli olarak uygulanır.

:::uyarı Kaynak düzeltmesi (lect_09): "bölümlü tarama" başka bir şeydir
Slaytlar bu hiyerarşik yöntemi "segmented parallel scan" diye adlandırıyor. Paralel algoritmalar literatüründe **bölümlü tarama** (segmented scan, Blelloch 1990) farklı bir işlemdir. Girdiyle birlikte bir bayrak dizisi verilir ve tarama her bayrakta **yeniden başlar**: birden çok bağımsız dizinin taramasını tek bir paralel taramayla yapar. İşleç, (değer, bayrak) çiftleri üzerinde birleşmeli bir işleme dönüştürülerek elde edilir. Bölümleri birleştiren yukarıdaki yöntemin yaygın adı **hiyerarşik** (ya da blok, "reduce-then-scan") taramadır.
:::

### Tek geçişli tarama {#tek-gecis}

Hiyerarşik taramada kısmi sonuçlar (bölüm taramaları) global belleğe yazılır, sonra üçüncü çekirdekte yeniden okunur. Bu ek saklama ve yüklemeler sonraki çekirdeklerin hesabıyla örtüşmez ve hızı belirgin etkiler. Birkaç teknik önerilmiştir (Dotsenko ve ark. 2008, Yan ve ark. 2013, Merrill ve Garland 2016). Bunlardan biri **akış tabanlı** ya da domino tarzı taramadır; CUDA akışlarıyla karıştırılmamalıdır. Kısmi toplam, aynı griddeki komşu bloklar arasında global bellek üzerinden tek yönde aktarılır. Temel gözlem, küresel tarama adımının domino gibi yapılabileceği ve gerçekten bütün grid için bir eşzamanlama gerektirmediğidir.

Tek bir çekirdek üç adımı birden yapar:

1. Blok $i$ bölümünü tarar. Bu evrede bütün bloklar paralel çalışır.
2. Blok $i$, sol komşusu $i-1$'in toplamı aktarmasını bekler, alınca kendi yerel toplamına ekler ve sağ komşusu $i+1$'e aktarır. Bu evre bloklar arasında sıralıdır.
3. Blok $i$, aldığı toplamı bölümünün bütün öğelerine ekler. Bir blok önceki bloğun değerini alır almaz bu evreyi diğerleriyle paralel yürütür. Toplamlar hızla aktarıldığı sürece üçüncü evrede bloklar arasında bol paralellik vardır.

Bunun için **komşu blok eşzamanlaması** gerekir. Üretici (blok $i-1$) kısmi toplamı belleğe yazdıktan sonra bir **bayrak** koyar. Tüketici (blok $i$) bayrağı denetler ve değerini görünce toplamı okur. Bu, M10.2'deki üretici–tüketici zincirinin GPU sürümüdür ve atomik işlemlerle gerçekleştirilir. Her bloğun yalnız bir lider thread'i (ör. thread 0) bu işi yapar:

```text
if (threadIdx.x == 0) {
  while (atomicAdd(&flags[bid], 0) == 0) { }      // sol komşunun bayrağını bekle
  previous_sum = scan_value[bid];
  scan_value[bid + 1] = previous_sum + local_sum;
  __threadfence();                                // değer, bayraktan önce görünür olsun
  atomicAdd(&flags[bid + 1], 1);
}
__syncthreads();
```

Diğer thread'ler sondaki `__syncthreads()`'te bekler. `__threadfence()`, `scan_value[bid + 1]` değerinin bayrak atomik işlemle konmadan önce global belleğe ulaşmasını güvence eder. Bu, M10.2'deki release semantiğinin karşılığıdır. Bayrak ve değer erişimleri global bellek trafiği gibi görünse de modern GPU'larda çoğunlukla L2 önbelleğinde kalır ve diğer blokların 1. ve 3. evre hesaplarıyla örtüşür. Üç çekirdekli sürümde $S$ dizisine yapılan saklama ve yüklemeler ise ayrı bir çekirdektedir ve örtüşemez.

:::uyarı Domino taramasında kilitlenme: blok sırası
Domino taramasının ince bir tuzağı vardır. Blok $i$, blok $i-1$'i bekler; ama CUDA blokların hangi sırayla zamanlanacağını güvence etmez (M10.4). Blok $i$ bir SM'de dönerek beklerken blok $i-1$ henüz hiç başlamamış olabilir. Bütün SM'ler bekleyen bloklarla doluysa blok $i-1$ asla zamanlanamaz: **kilitlenme**. Çözüm, blok indeksini `blockIdx.x`'ten değil, blok başladığında küresel bir sayaçtan atomik olarak almaktır: `bid = atomicAdd(&counter, 1)`. Böylece $i$ numaralı blok ancak $i-1$ numaralı blok başladıktan sonra var olur. Merrill ve Garland'ın "decoupled look-back" yöntemi bunu ve beklemeyi daha da iyileştirir: bir blok, yalnız soldaki komşusunu değil, toplamı hazır olan en yakın önceki bloğu arar. CUB ve Thrust'ın taramaları bu yöntemi kullanır.
:::

### Uygulamalar {#uygulamalar}

- **Akış sıkıştırma** (stream compaction): bir koşulu sağlayan öğeleri yeni bir diziye toplamak. Önce koşul bayrakları (0/1) hesaplanır, sonra **dışlayıcı taraması** alınır: her seçilen öğenin hedef konumu taramadaki değeridir (Şekil 2). Yazmalar bağımsızdır ve sıra korunur. GPU'da filtreleme, seyrek veri üretme ve iş listesi kurma bu yolla yapılır.
- **Taban sıralaması** (radix sort): her bit (ya da bit grubu) için bir sıkıştırma. 0 bitli öğeler önce, 1 bitliler sonra yazılır; hedefler bayrakların taramasıdır. $b$ bitlik anahtarlar $b$ geçişte sıralanır. Her geçiş kararlıdır, dolayısıyla sonuç doğru sıralıdır.
- **Sayma sıralaması ve histogram yerleşimi:** kutu sayıları tarandığında her kutunun çıktıdaki başlangıç konumu bulunur.
- **İntegral görüntü:** satırlarda, sonra sütunlarda iki tarama (M9.16, M10.3).
- **Bölümlü tarama:** bayraklı taramayla birçok küçük dizi tek seferde taranır.
- **Doğrusal özyinelemeler:** aşağıda.

<figure class="fig wide"><svg viewBox="0 0 640 206" role="img" aria-label="Dışlayıcı tarama ile akış sıkıştırma: sıfır olmayan öğelerin paralel olarak yeni bir diziye toplanması"><text x="160" y="38" text-anchor="end" class="sm">girdi x</text><rect x="170" y="20" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="191.0" y="38" text-anchor="middle" class="sm">3</text><rect x="216" y="20" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="237.0" y="38" text-anchor="middle" class="sm">0</text><rect x="262" y="20" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="283.0" y="38" text-anchor="middle" class="sm">7</text><rect x="308" y="20" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="329.0" y="38" text-anchor="middle" class="sm">0</text><rect x="354" y="20" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="375.0" y="38" text-anchor="middle" class="sm">0</text><rect x="400" y="20" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="421.0" y="38" text-anchor="middle" class="sm">2</text><rect x="446" y="20" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="467.0" y="38" text-anchor="middle" class="sm">5</text><rect x="492" y="20" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="513.0" y="38" text-anchor="middle" class="sm">0</text><text x="160" y="72" text-anchor="end" class="sm">bayrak (x ≠ 0)</text><rect x="170" y="54" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="191.0" y="72" text-anchor="middle" class="sm">1</text><rect x="216" y="54" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="237.0" y="72" text-anchor="middle" class="sm">0</text><rect x="262" y="54" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="283.0" y="72" text-anchor="middle" class="sm">1</text><rect x="308" y="54" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="329.0" y="72" text-anchor="middle" class="sm">0</text><rect x="354" y="54" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="375.0" y="72" text-anchor="middle" class="sm">0</text><rect x="400" y="54" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="421.0" y="72" text-anchor="middle" class="sm">1</text><rect x="446" y="54" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="467.0" y="72" text-anchor="middle" class="sm">1</text><rect x="492" y="54" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="513.0" y="72" text-anchor="middle" class="sm">0</text><text x="160" y="106" text-anchor="end" class="sm">dışlayıcı tarama</text><rect x="170" y="88" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="191.0" y="106" text-anchor="middle" class="sm">0</text><rect x="216" y="88" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="237.0" y="106" text-anchor="middle" class="sm">1</text><rect x="262" y="88" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="283.0" y="106" text-anchor="middle" class="sm">1</text><rect x="308" y="88" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="329.0" y="106" text-anchor="middle" class="sm">2</text><rect x="354" y="88" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="375.0" y="106" text-anchor="middle" class="sm">2</text><rect x="400" y="88" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="421.0" y="106" text-anchor="middle" class="sm">2</text><rect x="446" y="88" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="467.0" y="106" text-anchor="middle" class="sm">3</text><rect x="492" y="88" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="513.0" y="106" text-anchor="middle" class="sm">4</text><text x="160" y="160" text-anchor="end" class="sm">çıktı</text><rect x="170" y="142" width="42" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="191.0" y="160" text-anchor="middle" class="sm">3</text><rect x="216" y="142" width="42" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="237.0" y="160" text-anchor="middle" class="sm">7</text><rect x="262" y="142" width="42" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="283.0" y="160" text-anchor="middle" class="sm">2</text><rect x="308" y="142" width="42" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="329.0" y="160" text-anchor="middle" class="sm">5</text><path d="M191.0,114 L191.0,142" class="ca" stroke-dasharray="3 2"/><path d="M283.0,114 L237.0,142" class="ca" stroke-dasharray="3 2"/><path d="M421.0,114 L283.0,142" class="ca" stroke-dasharray="3 2"/><path d="M467.0,114 L329.0,142" class="ca" stroke-dasharray="3 2"/><text x="20" y="192" class="sm">Her thread kendi öğesini, bayrağı 1 ise tarama değeri kadar ilerideki konuma yazar: out[ex[i]] = x[i]. Çıktı boyu = son tarama + son bayrak = 4.</text></svg><figcaption>Şekil 2 — Akış sıkıştırma (stream compaction): koşulu sağlayan öğelerin hedef konumları, bayrakların dışlayıcı taramasıdır. Yazmalar birbirinden bağımsızdır ve sırayı korur (hesaplandı).</figcaption></figure>

### Doğrusal özyineleme ve durum uzay modelleri {#ozyineleme}

$x_t=a_tx_{t-1}+b_t$ biçimindeki bir özyineleme sıralı görünür, ama M9.18'de gördüğümüz gibi birleşmeli bir işleçle taranabilir. Her adım bir afin fonksiyondur, $f_t(x)=a_tx+b_t$. Bileşim
$$(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,\;a_2b_1+b_2)$$
birleşmelidir. Bu işleçle (değişmeli değildir; sıra korunmalıdır) Kogge–Stone ya da Brent–Kung taraması bütün $x_t$'leri $O(\log T)$ adımda üretir. Üstel hareketli ortalama (M9.18), filtreler (IIR) ve **durum uzay modelleri** (S4, Mamba) bu biçimdedir. Mamba'nın girdiye bağlı $\bar A_t$, $\bar B_t$ katsayılarıyla yinelemesi GPU'da tam olarak böyle bir paralel taramayla eğitilir.

**Kapalı biçim neden kullanılmaz?** $P_t=\prod_{k\le t}a_k$ ile $x_t=P_t\bigl(x_0+\sum_{k\le t}b_k/P_k\bigr)$ yazılabilir. Bu bir kümülatif çarpım ve bir kümülatif toplamdır; ikisi de taranabilir. Ama $|a_t|<1$ ise $P_t$ üstel olarak küçülür. 1024 adımda $0.9^{1024}\approx10^{-47}$ olur ve fp32'nin en küçük normal sayısının ($1.2\cdot10^{-38}$) altına düşer. $1/P_k$ taşar ve sonuç çöker. Afin bileşim taraması ise yalnız yakın geçmişin çarpımlarını taşır ve kararlıdır. Aşağıdaki deney bunu ölçüyor. Varsayılan ayarda (1024 adım, $a\approx0.9$) sıralı döngünün ve taramanın göreli hatası $\sim10^{-7}$, kümülatif çarpım yöntemi ise taşıyor.

::lab l2

### Özet {#ozet}

- Tarama (önek toplamı) birleşmeli bir işleç ve birim eleman ister. Kapsayıcı ve dışlayıcı biçimleri vardır. Sıralı tarama $n-1$ işlem yapar.
- Naif paralel tarama $O(n^2)$ iş yapar ve hızlanma vermez.
- Kogge–Stone $\log_2n$ adımda $n\log_2n-(n-1)$ işlem yapar (512 öğede 8 kat iş). Yazma-sonrası-okuma tehlikesi geçici değişken ve ikinci bariyerle ya da çift tamponla önlenir.
- Brent–Kung yukarı süpürme ve aşağı süpürmeden oluşur: $2\log_2n-1$ adımda $2n-2-\log_2n$ işlem (en çok 2 kat). Thread eşlemesi ayrışmayı azaltır.
- Hiyerarşik tarama üç adımdır: bölüm taramaları, blok toplamlarının taraması, ofset ekleme. Bu yöntem bayraklı "bölümlü taramadan" farklıdır.
- Tek geçişli domino taraması komşu blok eşzamanlamasıyla (bayrak, atomik, `__threadfence`) çalışır. Blok indeksi atomik sayaçtan alınmazsa kilitlenebilir; decoupled look-back bunu iyileştirir.
- Uygulamalar: akış sıkıştırma (dışlayıcı tarama), taban ve sayma sıralaması, integral görüntü, bölümlü tarama, doğrusal özyineleme ve durum uzay modelleri. Kümülatif çarpımlı kapalı biçim fp32'de taşar, afin bileşim taraması kararlıdır.
