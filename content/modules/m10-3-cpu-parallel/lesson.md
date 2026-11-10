### Alıştırmaların çerçevesi {#cerceve}

BİL 455'in ilk iki alıştırması aynı kalıbı izler: bir çekirdeği önce **tek thread'le**, sonra standart C++ thread'leriyle yaz. Ardından thread sayısını (4, 8, sistemin izin verdiği en çok) ve problem boyunu değiştirerek süreleri karşılaştır. Problem boyları 4096 tamsayıdan milyonlara, 100 × 100'lük görüntüden büyük görüntülere uzanır. Çekirdekler beş tanedir:

1. 8 bitlik gri görüntünün histogramı (Ex. 1.1). Bir sürüm yalnız atomik türlerle, mutex'siz; bir sürüm yalnız mutex'lerle, atomiksiz yazılır.
2. 32 bitlik tamsayı dizisinin integral dizisi, yani önek toplamı (Ex. 1.2).
3. 8 bitlik gri görüntünün integral görüntüsü (Ex. 1.3).
4. 2-B evrişim (Ex. 2.1). Girdi uint8, katsayılar fp32, bütün hesap fp32, çıktı uint8. Filtre boyları 3 × 3, 7 × 7, 11 × 11 ve 21 × 21'dir; Gauss gibi merkezi ağırlıklı çekirdekler kullanılır.
5. Matris çarpımı (Ex. 2.2): 32 bitlik tamsayı ya da fp32 girdi, fp32 hesap, rastgele matrisler. Sonuç bir doğrusal cebir kütüphanesiyle sınanır.

Görüntüler OpenCV'nin C++ API'siyle okunur; burada aynı denetimleri PyTorch'la yapıyoruz. Bu modül, her çekirdekte işin nasıl bölündüğünü (geometrik ayrıştırma, M10.1), ardışık bağımlılıkların nasıl kırıldığını ve hızlanmanın thread sayısına, problem boyuna ve belleğe nasıl bağlı olduğunu türetiyor. Thread araçları M10.2'den gelir.

### Doğru ölçmek {#olcum}

Karşılaştırmaların anlamlı olması için M10.1'in altın kuralları uygulanır:

- **Aynı koşullar, en hızlı sıralı kod.** İkisi de eniyileme açık (`-O2`/`-O3`) derlenir. `-O0` ile ölçülen hızlanma anlamsızdır: eniyilenmemiş sıralı kod yapay biçimde yavaştır.
- **Doğru saat.** `std::chrono::steady_clock` kullanılır; `system_clock` ayarlanabilir.
- **Isınma ve tekrar.** Bir ısınma çalıştırması yapılır, ölçüm birkaç kez (örneğin 5) tekrarlanıp ortanca alınır. İlk çalıştırma sayfa hatalarını ve önbelleği ödediği için yavaştır.
- **Thread sayısı.** "Sistemin izin verdiği en çok thread" için `std::thread::hardware_concurrency()` mantıksal çekirdek sayısını verir. Eşzamanlı çoklu iş parçacığı (SMT, hiper iş parçacığı) olan makinelerde fiziksel çekirdek sayısının ötesinde kazanç küçüktür, çünkü iki mantıksal çekirdek aynı hesap birimlerini paylaşır.
- **Ne ölçüldüğü.** Thread yaratma süresi ölçüme dahil mi? İkisi de bildirilmelidir. Bir thread'i yaratıp `join` etmek tipik olarak 10–50 µs sürer; bu, 4096 elemanlık bir önek toplamının kendisinden büyüktür.
- **Doğruluk.** Paralel sonuç sıralı sonuçla (tamsayıda birebir, kayan noktada göreli hata içinde) karşılaştırılmadan hiçbir hızlanma bildirilmez.

### Histogram {#histogram}

Tek thread'li histogram basittir: `for (p : pixels) h[p]++;`. Çok thread'li sürümde pikseller thread'lere geometrik olarak bölünür (her thread ardışık bir blok), ama **histogram paylaşılır**. M10.2'deki kayıp güncelleme sorunu burada birebir ortaya çıkar: iki thread aynı kutuyu aynı anda artırırsa biri kaybolur. Alıştırmanın üç sürümü vardır:

1. **Paylaşılan histogram + `std::atomic<int>` kutular, mutex yok:** `h[p].fetch_add(1, memory_order_relaxed)`. Doğrudur. Sıra önemsiz olduğu için relaxed yeter, sonunda `join` zaten eşzamanlar.
2. **Paylaşılan histogram + mutex, atomik yok:** her artırma bir kilit içinde. Doğrudur, ama kritik bölge her pikselde girildiği için thread'ler sıraya girer.
3. **Thread'e özel histogramlar + birleştirme:** her thread kendi 256 kutusunu doldurur, sonunda bunlar toplanır. Bu, M10.2'deki "yerel kopya + tek kilitli birleştirme" kalıbıdır.

:::türetim Atomik histogram neden ölçeklenmez?
256 tane 4 baytlık kutu 1 KB eder, yani 64 baytlık 16 önbellek satırı. Bir çekirdeğin bir satıra yazabilmesi için satırın o çekirdekte, "değiştirilmiş" durumda olması gerekir (MESI). Satır başka bir çekirdekteyse oradan istenir ve öteki kopya geçersizleştirilir; bu bir **satır transferidir** ve onlarca–yüzlerce çevrim sürer.

$p$ thread pikselleri eşzamanlı işlerken bir kutunun satırını en son başka bir thread'in yazmış olma olasılığı yaklaşık $1-1/p$'dir. Dolayısıyla neredeyse her artırma bir transfer gerektirir. Üstelik bir satır aynı anda tek bir çekirdekte bulunabilir: aynı satıra yazan çekirdekler sıraya girer. Histogram dar tepeliyse (bütün pikseller birkaç değerde) işin tamamı tek bir satırda serileşir.

Kilitli artırmanın (`lock xadd`) kendi maliyeti de vardır: transfer olmasa bile yaklaşık 20 çevrim, sıradan bir artırmanın on katından fazla.

Özel histogramlarda paylaşım yoktur: her artırma kendi önbelleğinde yerel bir işlemdir. Birleştirmenin maliyeti $256\,p$ toplamadır; 4 thread'le 1024. $N\gg256p$ olduğu sürece ihmal edilebilir. Hızlanma yaklaşık $p$'dir ve histogramın biçimine bağlı değildir.
:::

Aşağıdaki deney bu maliyet modelini simüle ediyor; sayılar ölçüm değil, çevrim cinsinden bir modeldir. Doğal bir görüntü histogramı ve 8 thread'le:

- **Atomik:** tek thread'den 8.5 kat **yavaş** (hızlanma 0.12; piksellerin %92'si satır transferi gerektiriyor); dar tepeli histogramda 37 kat yavaş.
- **Küresel mutex:** 20 kat yavaş.
- **Özel histogram:** 7.1 kat hızlı.

Gerçek makinelerde de atomik ve mutex'li histogramların tek thread'den yavaş çıkması yaygın bir ölçümdür.

::lab l1

### İntegral dizi: önek toplamı {#onek}

Integral dizi $S_i=\sum_{j\le i}a_j$'dir. Sıralı algoritma tek döngüdür: `S[0] = a[0]; for (i = 1; i < N; i++) S[i] = S[i-1] + a[i];`. Her eleman bir öncekine bağlıdır; döngüyü thread'lere bölmenin dolaysız bir yolu yoktur. Bağımlılık **blok toplamlarıyla** kırılır (Şekil 1):

1. **Yerel tarama:** dizi $p$ bloğa bölünür, her thread kendi bloğunu yerel olarak tarar.
2. **Blok toplamlarının taranması:** her bloğun toplamı (yerel taramanın son elemanı) alınır. Bu $p$ sayının dışlayıcı taraması, yani her bloğun **ofseti** (önceki blokların toplamı), tek thread'de hesaplanır.
3. **Ofset ekleme:** her thread bloğunun her elemanına kendi ofsetini ekler.

İlk ve üçüncü adım arasında bir bariyer gerekir (thread'leri `join` edip yeniden başlatmak ya da bir `std::barrier`).

:::türetim Doğruluk, iş ve yol
Blok $b$'deki bir $i$ indeksi için $S_i=\sum_{j<\text{başlangıç}_b}a_j+\sum_{\text{başlangıç}_b\le j\le i}a_j$ yazılabilir. İlk terim önceki blokların toplamı, yani ofsettir. İkinci terim yerel taramadır. Bu kadar basit bir ayrışım, toplamanın **birleşmeli** olmasına dayanır: $(a+b)+c=a+(b+c)$. Bu yüzden aynı yöntem herhangi bir birleşmeli işlem için çalışır (M10.6).

$[1,\dots,8]$ ve 4 thread için blok toplamları $(3,7,11,15)$, ofsetler $(0,3,10,21)$'dir. 3. bloğun (5, 6) toplamı 11, ofseti 10'dur; yerel tarama $(5,11)$ ofsetle $(15,21)$ olur.

İş yaklaşık $2N$ toplamadır (sıralı algoritmanın iki katı), yol $2N/p+p$. Paralel algoritma **daha fazla iş** yapar. Hesap sınırlıysa $p$ çekirdekte hızlanma en çok yaklaşık $p/2$'dir.
:::

**Topla, sonra tara.** Bir değişik sürüm birinci adımda bloğu taramaz, yalnız **toplar** (salt okuma). Üçüncü adımda bloğu ofsetten başlayarak tarar. İş yine $2N$'dir, ama bellek trafiği farklıdır:

- **Tara–düzelt:** iki geçişin ikisi de diziyi okuyup yazar; $4N$ eleman trafiği.
- **Topla–tara:** ilk geçiş yalnız okur; $3N$ eleman trafiği.

Dizi önbelleğe sığmadığında süreyi hesap değil **bellek bant genişliği** belirler. Bir çekirdek ana belleğin bant genişliğinin yalnız bir kısmını doldurabilir; birkaç çekirdekten sonra bant genişliği doyar. Hızlanma bu tavanı aşamaz ve trafiği azaltan topla–tara sürümü daha iyidir.

:::uyarı Taşma: 32 bitlik toplamlar
Alıştırma 32 bitlik `int` dizisi varsayar, ama önek toplamları girdiden çok daha büyük olur. 4 milyon elemanın her biri 1000'e kadarsa son toplam $4\cdot10^9$'a çıkar ve $2^{31}-1\approx2.1\cdot10^9$'u aşar. İşaretli taşma C++'ta tanımsız davranıştır. Çıktı 64 bitlik (`int64_t`) tutulmalı ya da girdi aralığı buna göre sınırlanmalıdır. Aynı sorun integral görüntüde de vardır (aşağıda).
:::

Problem boyunun etkisi belirleyicidir. 4096 elemanlık bir dizinin sıralı taraması birkaç mikrosaniye sürer; 8 thread yaratmak ise onlarca mikrosaniye. Aşağıdaki deneyin maliyet modelinde (thread yaratma 20 000 çevrim):

- 8 thread 4096 elemanda tek thread'den 40 kat yavaş.
- Paralel sürüm ancak $2^{18}$ elemandan sonra kazandırıyor.
- Önbelleğe sığmayan dizilerde hızlanma bant genişliği tavanında doyuyor: tara–düzelt 3.0, topla–tara 4.0.

Thread havuzu (M10.2) yaratma maliyetini bir kez öder ve başa baş noktasını küçültür.

::lab l2

### İntegral görüntü {#integral}

İntegral görüntü $I(y,x)=\sum_{y'\le y,\,x'\le x}f(y',x')$'tir (M9.16). Sıralı algoritma iki geçişle yazılabilir:

1. Her satırın önek toplamı: $R(y,x)=\sum_{x'\le x}f(y,x')$.
2. Her sütunda bu değerlerin önek toplamı: $I(y,x)=\sum_{y'\le y}R(y',x)$.

Tek geçişli biçim $I(y,x)=I(y-1,x)+R(y,x)$'tir. Paralellik iki geçişten gelir:

- **Satır geçişinde** satırlar birbirinden bağımsızdır: thread'ler satır bantlarına bölünür.
- **Sütun geçişinde** sütunlar bağımsızdır, ama sütun boyunca yürümek önbellek dostu değildir: ardışık erişimler $W$ eleman arayla düşer.

Doğru yol, sütun geçişini de satır satır yapmaktır. Her thread bir **sütun şeridi** $[x_0,x_1)$ alır ve satırları yukarıdan aşağı tarar: `I[y][x] = I[y-1][x] + R[y][x]`, $x\in[x_0,x_1)$. Her satırda şeridin içi ardışıktır. İki geçiş arasında bir bariyer gerekir.

<figure class="fig wide"><svg viewBox="0 0 640 384" role="img" aria-label="İki geçişli paralel blok taraması ve integral görüntünün satır ve sütun geçişleri, sayısal örnekle"><text x="140" y="38" text-anchor="end" class="sm">girdi a</text><rect x="150" y="20" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="171.0" y="38" text-anchor="middle" class="sm">1</text><rect x="196" y="20" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="217.0" y="38" text-anchor="middle" class="sm">2</text><rect x="242" y="20" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="263.0" y="38" text-anchor="middle" class="sm">3</text><rect x="288" y="20" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="309.0" y="38" text-anchor="middle" class="sm">4</text><rect x="334" y="20" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="355.0" y="38" text-anchor="middle" class="sm">5</text><rect x="380" y="20" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="401.0" y="38" text-anchor="middle" class="sm">6</text><rect x="426" y="20" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="447.0" y="38" text-anchor="middle" class="sm">7</text><rect x="472" y="20" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="493.0" y="38" text-anchor="middle" class="sm">8</text><text x="140" y="74" text-anchor="end" class="sm">1: yerel tarama</text><rect x="150" y="56" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="171.0" y="74" text-anchor="middle" class="sm">1</text><rect x="196" y="56" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="217.0" y="74" text-anchor="middle" class="sm">3</text><rect x="242" y="56" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="263.0" y="74" text-anchor="middle" class="sm">3</text><rect x="288" y="56" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="309.0" y="74" text-anchor="middle" class="sm">7</text><rect x="334" y="56" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="355.0" y="74" text-anchor="middle" class="sm">5</text><rect x="380" y="56" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="401.0" y="74" text-anchor="middle" class="sm">11</text><rect x="426" y="56" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="447.0" y="74" text-anchor="middle" class="sm">7</text><rect x="472" y="56" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="493.0" y="74" text-anchor="middle" class="sm">15</text><text x="140" y="110" text-anchor="end" class="sm">2: blok toplamları</text><rect x="150" y="92" width="88" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="194" y="110" text-anchor="middle" class="sm">3</text><rect x="242" y="92" width="88" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="286" y="110" text-anchor="middle" class="sm">7</text><rect x="334" y="92" width="88" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="378" y="110" text-anchor="middle" class="sm">11</text><rect x="426" y="92" width="88" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="470" y="110" text-anchor="middle" class="sm">15</text><text x="140" y="146" text-anchor="end" class="sm">   ofsetler (tarama)</text><rect x="150" y="128" width="88" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="194" y="146" text-anchor="middle" class="sm">0</text><rect x="242" y="128" width="88" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="286" y="146" text-anchor="middle" class="sm">3</text><rect x="334" y="128" width="88" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="378" y="146" text-anchor="middle" class="sm">10</text><rect x="426" y="128" width="88" height="26" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="470" y="146" text-anchor="middle" class="sm">21</text><text x="140" y="182" text-anchor="end" class="sm">3: ofset ekle</text><rect x="150" y="164" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="171.0" y="182" text-anchor="middle" class="sm">1</text><rect x="196" y="164" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="217.0" y="182" text-anchor="middle" class="sm">3</text><rect x="242" y="164" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="263.0" y="182" text-anchor="middle" class="sm">6</text><rect x="288" y="164" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="309.0" y="182" text-anchor="middle" class="sm">10</text><rect x="334" y="164" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="355.0" y="182" text-anchor="middle" class="sm">15</text><rect x="380" y="164" width="42" height="26" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="401.0" y="182" text-anchor="middle" class="sm">21</text><rect x="426" y="164" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="447.0" y="182" text-anchor="middle" class="sm">28</text><rect x="472" y="164" width="42" height="26" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="493.0" y="182" text-anchor="middle" class="sm">36</text><text x="20" y="206" class="sm">4 thread, blok başına 2 eleman. 1. ve 3. adım paralel (her thread kendi bloğu);</text><text x="20" y="222" class="sm">2. adım p elemanlık küçük bir sıralı tarama.</text><text x="150" y="246" class="sm">f</text><rect x="150" y="252" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="163.0" y="270" text-anchor="middle" class="sm">1</text><rect x="178" y="252" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="191.0" y="270" text-anchor="middle" class="sm">2</text><rect x="206" y="252" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="219.0" y="270" text-anchor="middle" class="sm">0</text><rect x="150" y="280" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="163.0" y="298" text-anchor="middle" class="sm">3</text><rect x="178" y="280" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="191.0" y="298" text-anchor="middle" class="sm">4</text><rect x="206" y="280" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="219.0" y="298" text-anchor="middle" class="sm">1</text><rect x="150" y="308" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="163.0" y="326" text-anchor="middle" class="sm">0</text><rect x="178" y="308" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="191.0" y="326" text-anchor="middle" class="sm">2</text><rect x="206" y="308" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="219.0" y="326" text-anchor="middle" class="sm">5</text><text x="300" y="246" class="sm">satır taraması</text><rect x="300" y="252" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="313.0" y="270" text-anchor="middle" class="sm">1</text><rect x="328" y="252" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="341.0" y="270" text-anchor="middle" class="sm">3</text><rect x="356" y="252" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="369.0" y="270" text-anchor="middle" class="sm">3</text><rect x="300" y="280" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="313.0" y="298" text-anchor="middle" class="sm">3</text><rect x="328" y="280" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="341.0" y="298" text-anchor="middle" class="sm">7</text><rect x="356" y="280" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="369.0" y="298" text-anchor="middle" class="sm">8</text><rect x="300" y="308" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="313.0" y="326" text-anchor="middle" class="sm">0</text><rect x="328" y="308" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="341.0" y="326" text-anchor="middle" class="sm">2</text><rect x="356" y="308" width="26" height="26" class="soft-a" stroke="currentColor" stroke-width="0.5"/><text x="369.0" y="326" text-anchor="middle" class="sm">7</text><text x="450" y="246" class="sm">sütun taraması = I</text><rect x="450" y="252" width="26" height="26" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="463.0" y="270" text-anchor="middle" class="sm">1</text><rect x="478" y="252" width="26" height="26" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="491.0" y="270" text-anchor="middle" class="sm">3</text><rect x="506" y="252" width="26" height="26" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="519.0" y="270" text-anchor="middle" class="sm">3</text><rect x="450" y="280" width="26" height="26" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="463.0" y="298" text-anchor="middle" class="sm">4</text><rect x="478" y="280" width="26" height="26" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="491.0" y="298" text-anchor="middle" class="sm">10</text><rect x="506" y="280" width="26" height="26" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="519.0" y="298" text-anchor="middle" class="sm">11</text><rect x="450" y="308" width="26" height="26" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="463.0" y="326" text-anchor="middle" class="sm">4</text><rect x="478" y="308" width="26" height="26" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="491.0" y="326" text-anchor="middle" class="sm">12</text><rect x="506" y="308" width="26" height="26" class="soft-p" stroke="currentColor" stroke-width="0.5"/><text x="519.0" y="326" text-anchor="middle" class="sm">18</text><text x="20" y="292" class="sm">integral</text><text x="20" y="308" class="sm">görüntü</text><text x="20" y="356" class="sm">Satır geçişinde satırlar, sütun geçişinde sütunlar birbirinden bağımsız;</text><text x="20" y="372" class="sm">iki geçiş arasında bir bariyer (join).</text></svg><figcaption>Şekil 1 — Önek toplamındaki ardışık bağımlılık blok toplamlarıyla kırılır: [1…8] için 3. bloğun toplamı 11, ofseti 10; integral görüntü iki bağımsız tarama geçişidir (I(2, 2) = 18; hesaplandı).</figcaption></figure>

İntegral görüntünün değerleri hızla büyür. $H\times W$'lik 8 bitlik bir görüntüde en büyük değer $255HW$'dir. 100 × 100'de 2.55 milyon, 32 bite rahatça sığar. 4096 × 4096'da ise $255\cdot4096^2=4.28\cdot10^9$ olur: işaretli 32 bitlik tamsayıyı aşar, işaretsize ancak sığar. Daha büyük görüntülerde 64 bit gerekir.

### 2-B evrişim {#evrisim}

Alıştırmanın kuralları veri türlerini belirler. Pikseller uint8 okunur, fp32'ye çevrilir, fp32 katsayılarla çarpılıp toplanır ve sonuç yuvarlanıp $[0,255]$'e kırpılarak uint8'e geri yazılır. Kırpma unutulursa, 255'i aşan değerler uint8'e dönüşürken sarmalanır (300 → 44): parlak bölgelerde koyu lekeler belirir.

Toplam iş $HWk^2$ çarp-topladır: 512 × 512'de 21 × 21 filtreyle $115\,605\,504$. **Geometrik ayrıştırma** doğaldır: çıktı görüntüsü $p$ satır bandına bölünür ve her thread kendi bandını hesaplar. Bir çıktı pikseli girdinin $k\times k$ komşuluğunu okuduğu için, $h$ satırlık bir bant girdinin $h+2\lfloor k/2\rfloor$ satırını okur. Fazladan okunan satırlara **hale** (halo) denir: 21 × 21 için üstte ve altta 10'ar, toplam 20 satır. Hale **salt okunurdur**: thread'ler yalnız kendi bantlarına yazar, yarış yoktur ve eşzamanlama gerekmez. Kenarlarda görüntünün dışına taşan komşuluk için bir sınır koşulu seçilir: sıfır dolgu, kenarı tekrarlama ya da yansıtma (M9.3).

<figure class="fig wide"><svg viewBox="0 0 640 230" role="img" aria-label="2-B evrişimin satır bantlarına ayrıştırılması, bir bandın hale bölgesi ve filtre boyuna göre maliyet tablosu"><rect x="20" y="20.0" width="184.3" height="46.1" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="112.2" y="47.0" text-anchor="middle" class="sm">thread 0</text><rect x="20" y="66.1" width="184.3" height="46.1" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="112.2" y="93.1" text-anchor="middle" class="sm">thread 1</text><rect x="20" y="112.2" width="184.3" height="46.1" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="112.2" y="139.2" text-anchor="middle" class="sm">thread 2</text><rect x="20" y="158.2" width="184.3" height="46.1" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="112.2" y="185.3" text-anchor="middle" class="sm">thread 3</text><rect x="20" y="62.5" width="184.3" height="53.3" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="4 3"/><text x="20" y="220.3" class="sm">thread 1'in okuduğu bölge (21 × 21: ± 10 satır hale)</text><text x="230" y="34" class="sm">k × k</text><text x="300" y="34" class="sm">MAC/piksel</text><text x="400" y="34" class="sm">ayrılabilir</text><text x="490" y="34" class="sm">hale payı (8 bant)</text><text x="230" y="58" class="sm">3 × 3</text><text x="300" y="58" class="sm">9</text><text x="400" y="58" class="sm">6 (1.5×)</text><text x="490" y="58" class="sm">3.1% fazla okuma</text><text x="230" y="80" class="sm">7 × 7</text><text x="300" y="80" class="sm">49</text><text x="400" y="80" class="sm">14 (3.5×)</text><text x="490" y="80" class="sm">9.4% fazla okuma</text><text x="230" y="102" class="sm">11 × 11</text><text x="300" y="102" class="sm">121</text><text x="400" y="102" class="sm">22 (5.5×)</text><text x="490" y="102" class="sm">15.6% fazla okuma</text><text x="230" y="124" class="sm">21 × 21</text><text x="300" y="124" class="sm">441</text><text x="400" y="124" class="sm">42 (10.5×)</text><text x="490" y="124" class="sm">31.2% fazla okuma</text><text x="230" y="160" class="sm">512 × 512, 21 × 21: 115 605 504 çarp-topla; ayrılabilir 11 010 048.</text><text x="230" y="182" class="sm">Bant yüksekliği h = H/p; hale 2⌊k/2⌋ satır: oran 2⌊k/2⌋ / h.</text><text x="230" y="204" class="sm">Hale salt okunur: thread'ler yalnız kendi bandına yazar, yarış yok.</text></svg><figcaption>Şekil 2 — Geometrik ayrıştırma: her thread bir çıktı bandı hesaplar ve girdinin ± ⌊k/2⌋ satırını fazladan okur. Maliyet k² ile büyür; ayrılabilir filtreler bunu 2k'ya indirir (hesaplandı).</figcaption></figure>

**Ayrılabilir filtreler.** Gauss gibi bir çekirdek bir satır ve bir sütun vektörünün dış çarpımıysa ($w=uv^T$), evrişim iki 1-B geçişe ayrılır. Maliyet $k^2$ yerine $2k$ olur: 21 × 21 için 441 yerine 42, yani 10.5 kat az. İki geçiş arasında ara görüntü fp32 tutulmalıdır; ara sonucu uint8'e yuvarlamak hata biriktirir. Ayrılabilir olmayan büyük çekirdeklerde FFT tabanlı evrişim (M9.10) $O(HW\log HW)$ ile $k$'dan bağımsızdır.

**Hızlanma.** Evrişim hesap sınırlıdır: her girdi pikseli $k^2$ kez kullanılır ve önbellekte kalır. Bu yüzden thread sayısıyla iyi ölçeklenir. Küçük filtrede (3 × 3) ve küçük görüntüde ise thread yaratma ve bellek trafiği öne çıkar. Bantların eşit olması yük dengesi için yeterlidir, çünkü her pikselin maliyeti aynıdır (M10.1'deki Mandelbrot'un tersi).

### Matris çarpımı {#matmul}

$C=AB$, $n\times n$ matrislerle $n^3$ çarp-topla, yani $2n^3$ FLOP yapar: $n=1024$ için $2.15\cdot10^9$. Veri ise yalnız $3n^2$ elemandır. **Aritmetik yoğunluk** (FLOP / bellek trafiği), her eleman bir kez okunursa $2n^3/(3\cdot4n^2)=n/6$ FLOP/bayttır. 1024'te 170'tir; bu, belleğin değil hesabın sınırlaması için fazlasıyla yeterlidir. Ama bu yoğunluk **ancak verinin yeniden kullanılmasıyla** elde edilir. Saf bir döngü her çarpımda belleğe gidiyorsa yoğunluk birkaç FLOP/bayta düşer.

**Döngü sırası.** Üç iç içe döngünün sırası sonucu değiştirmez, erişim desenini değiştirir. Matrisler satır öncelikli saklanır.

| sıra | iç döngü | A | B | C |
|---|---|---|---|---|
| ijk | k | satır boyunca | **sütun boyunca** (adım $n$) | yazmaçta |
| ikj | j | yazmaçta | satır boyunca | satır boyunca |
| jki | i | **sütun boyunca** | yazmaçta | **sütun boyunca** |

Sütun boyunca yürümek her erişimde farklı bir önbellek satırına düşer. 64 baytlık satırdan yalnız 4 bayt kullanılır, kalan 60 bayt satır başka bir şeyle değiştirilmeden önce kullanılmayabilir. Çözümler şunlardır:

- **ikj sırası:** iç döngü B'nin ve C'nin satırlarında yürür; ardışık erişim derleyicinin vektörleştirmesine de izin verir.
- **Devrik B:** $B^T$ önceden bir kez hesaplanır ($O(n^2)$), sonra ijk iç döngüsü iki satırın iç çarpımıdır.
- **Karolama** (tiling, blocking): matrisler $b\times b$ karolara bölünür ve üç karo önbelleğe sığacak biçimde $b$ seçilir ($3b^2\cdot4$ bayt $\le$ önbellek). Her karo $b$ kez yeniden kullanılır; aritmetik yoğunluk $b$ ile orantılı büyür.

:::türetim Karolamanın kazancı
Karosuz ikj'de B'nin her satırı her $i$ için yeniden okunur. Önbellek bütün B'yi tutamıyorsa toplam B trafiği $n\cdot n^2=n^3$ eleman olur, yoğunluk sabit kalır.

$b\times b$ karolarla bir C karosunu bitirmek için $n/b$ çift A ve B karosu okunur: her biri $b^2$ eleman, toplam $2b^2(n/b)=2nb$ eleman. Bu, $2nb^2$ FLOP içindir. C karolarının sayısı $(n/b)^2$ olduğundan toplam trafik $\approx2n^3/b$ elemandır, yani karosuz sürümden $b/2$ kat az. Yoğunluk $\approx b/4$ FLOP/bayttır.

Üç karo önbelleğe sığmalıdır. 32 KB'lık L1 için $3b^2\cdot4\le32768$, yani $b\le52$; pratikte 32. Gerçek kütüphaneler (BLAS, oneDNN) çok düzeyli karolama (yazmaçlar, L1, L2, L3) ve vektör komutlarıyla tepe başarımın %80–90'ına ulaşır.
:::

Aşağıdaki deney gerçek bir erişim izini 16 KB'lık, 8 yollu bir LRU önbellekte simüle ediyor. $n=64$'te FMA başına ıska sayısı şöyledir:

- ijk: 0.15;
- ikj: 0.018;
- jki: 0.62;
- devrik B: 0.018;
- karolu ikj: 0.006.

Beklenmedik bir şey de var: ijk'de $n=64$, $n=68$'den 2.4 kat fazla ıska veriyor. 64 bir 2'nin kuvvetidir: bir sütundaki ardışık elemanlar 256 bayt aralıklı düşer ve önbelleğin aynı birkaç setine yığılır. 8 yol yetmez ve **çakışma ıskaları** oluşur. Gerçek kodlarda satır uzunluğunu biraz dolgulamak (örneğin 64 yerine 68 eleman) bu sorunu giderir.

::lab l3

**Paralellik.** C'nin satır blokları birbirinden bağımsızdır: thread $t$, C'nin $[tn/p,(t+1)n/p)$ satırlarını hesaplar, A'nın ilgili satırlarını ve B'nin tamamını okur. Yazma çakışması yoktur. B paylaşılır ama salt okunurdur; her çekirdek kendi önbelleğine kopyasını alır. Karolu sürümde iş birimi bir C karosudur ve karolar thread'lere döngüsel ya da dinamik dağıtılabilir.

**Doğruluk denetimi.** Kayan noktalı toplama birleşmeli değildir. Farklı döngü sıraları ve karolar toplamları farklı sırada yaptığı için sonuçlar son bitlerde farklılaşır. Karşılaştırma birebir değil göreli hatayla yapılır. fp32'de $n$ terimlik bir iç çarpımın hatası $\sim\sqrt n\,\epsilon$ ile $n\,\epsilon$ arasında büyür ($\epsilon\approx6\cdot10^{-8}$); $n=1024$ için $10^{-5}$ mertebesinde bir göreli tolerans makuldür.

### PyTorch'un kendi çekirdekleri {#pytorch-cekirdek}

| alıştırma | PyTorch karşılığı | not |
|---|---|---|
| histogram | `torch.bincount(x, minlength=256)`, `torch.histc` | CPU'da thread'e özel kopyalar |
| integral dizi | `torch.cumsum(x, 0, dtype=torch.int64)` | taşmaya karşı dtype |
| integral görüntü | `x.cumsum(0).cumsum(1)` | iki geçiş |
| 2-B evrişim | `F.conv2d(x[None, None].float(), w[None, None], padding=k // 2)` | oneDNN; çıktıda `round().clamp(0, 255).to(torch.uint8)` |
| matris çarpımı | `torch.mm(A, B)` | MKL/oneDNN SGEMM |
| thread sayısı | `torch.set_num_threads(p)`, `torch.get_num_threads()` | OpenMP havuzu |

Bu çekirdekler aynı fikirleri en iyi biçimde uygular: thread havuzu, karolama, vektör komutları. Ödevdeki C++ sürümlerinin ulaşabileceği tavanı gösterirler. Küçük problemlerde PyTorch'un da sabit bir çağrı ek yükü (mikrosaniyeler) vardır ve tek thread'li saf bir C++ döngüsü ondan hızlı olabilir.

### Özet {#ozet}

- Ölçüm kuralları: eniyileme açık derleme, `steady_clock`, ısınma ve ortanca, thread yaratma süresinin ayrıca belirtilmesi, sıralı sonuçla doğrulama.
- Histogramda paylaşılan atomik ya da kilitli sürümler önbellek satırı transferleri ve serileşme yüzünden tek thread'den yavaştır. Thread'e özel histogram + birleştirme yaklaşık $p$ kat hızlanır.
- Önek toplamı blok toplamlarıyla paralelleşir: yerel tarama, ofset taraması ve ofset ekleme. İş $2N$, yol $2N/p+p$'dir. Topla–tara bellek trafiğini $4N$'den $3N$'e indirir. Küçük $N$'de thread maliyeti baskındır, büyük $N$'de bant genişliği. Taşmaya dikkat: 32 bitlik önek toplamı kolayca taşar.
- İntegral görüntü iki bağımsız taramadır; sütun geçişi sütun şeritleriyle önbellek dostu yapılır. 4096 × 4096'da değerler $4.28\cdot10^9$'a çıkar.
- Evrişim satır bantlarına bölünür ve salt okunur bir hale ister ($2\lfloor k/2\rfloor$ satır). Maliyet $HWk^2$'dir; ayrılabilir filtrede $2HWk$. Çıktı yuvarlanıp kırpılmalıdır.
- Matmul'da $2n^3$ FLOP ve aritmetik yoğunluk $n/6$'dır. Döngü sırasının etkisini ijk, ikj, jki ve devrik B sıraları gösterir. Karolama trafiği $b/2$ kat azaltır. 2'nin kuvveti boyutlar çakışma ıskaları üretir. Satır blokları thread'lere dağıtılır ve sonuç göreli hatayla doğrulanır.
