### İndirgeme {#indirgeme}

**İndirgeme**, bir veri grubuna uygulanabilen genelleştirilmiş bir işlemdir. Girdi bir dizi (vektör, liste) değerdir; çıktı girdiyi "özetleyen" tek bir değerdir. İndirgeme bir **işleç** ister; ona $\oplus$ diyelim:

- **Birleşmeli:** $(a\oplus b)\oplus c=a\oplus(b\oplus c)$.
- **Değişmeli:** $a\oplus b=b\oplus a$ (aşağıda hangi algoritmaların gerçekten gerektirdiğini göreceğiz).
- **Birim elemanı** $I$ vardır: $a\oplus I=a$. Toplamada 0, çarpmada 1, en büyükte $-\infty$.

Örnekler: kayan noktalı bir dizinin toplamı (işleç $+$), en büyük değeri (işleç max), argmax (değer–indeks çiftleri üzerinde max), çarpım, mantıksal VE/VEYA.

**Sıralı indirgeme** birim elemanla başlar ve her öğeyi sırayla birleştirir: `acc = I; for (i = 0; i < N; i++) acc = acc ⊕ x[i];`. $N$ adım, $N$ işlem. Toplam için `sum = 0; for (...) sum += x[i];`.

### İndirgeme ağacı {#agac}

Birleşmelilik işlemlerin gruplanmasını serbest bırakır. Sekiz öğenin en büyüğünü bulurken ilk adımda dört çiftin max'ı paralel alınır: dört kısmi sonuç. İkinci adımda iki çiftin max'ı alınır: ilk dört ve son dört öğenin en büyükleri. Üçüncü adımda tek bir max işlemi sonucu verir. Bu desene **indirgeme ağacı** denir, çünkü yaprakları girdi öğeleri, kökü sonuç olan bir ağaca benzer. Ağaç veri yapılarıyla karıştırılmamalıdır: orada düğümler işaretçilerle ya da konumlarla bağlıdır. İndirgeme ağacında kenarlar kavramsaldır ve bir zaman adımındaki işlemlerden bir sonrakine bilgi akışını gösterir.

:::türetim İş, adım ve paralellik
$N$ girdi için ilk adımda $N/2$, ikincide $N/4$ işlem yapılır, ve böyle devam eder. Toplam işlem sayısı bir geometrik seridir:
$$\frac N2+\frac N4+\dots+1=N-1.$$
Bu, sıralı algoritmayla aynıdır: ağaç **iş verimlidir**. Adım sayısı $\log_2N$'dir; $N=1024$ için 10 adım. Sekiz öğede sıralı döngü 8 adım, ağaç 3 adım sürer: adım sayısında %62.5'lik azalma, yani $8/3=2.67$ kat hızlanma.

Bedeli, kaynak gereksinimidir. İlk adımda $N/2$ yürütme birimi gerekir (1024 için 512), son adımda yalnız biri. **Ortalama paralellik** toplam işin adım sayısına oranıdır, $(N-1)/\log_2N$. $N=1024$ için 102.3'tür, oysa en yüksek paralellik 512'dir (Şekil 2). Adımlar arasındaki bu değişkenlik, indirgeme ağaçlarını paralel donanım için zor bir desen yapar. M10.1'in diliyle iş $W=N-1$, yol $S_\infty=\log_2N$'dir.
:::

<figure class="fig wide"><svg viewBox="0 0 640 232" role="img" aria-label="1024 öğelik bir indirgeme ağacında her zaman adımındaki işlem sayısı ve ortalama paralellik"><path d="M60,190.0 L420,190.0" class="mute"/><text x="54" y="194.0" text-anchor="end" class="sm">1</text><path d="M60,152.2 L420,152.2" class="mute"/><text x="54" y="156.2" text-anchor="end" class="sm">4</text><path d="M60,114.4 L420,114.4" class="mute"/><text x="54" y="118.4" text-anchor="end" class="sm">16</text><path d="M60,76.7 L420,76.7" class="mute"/><text x="54" y="80.7" text-anchor="end" class="sm">64</text><path d="M60,38.9 L420,38.9" class="mute"/><text x="54" y="42.9" text-anchor="end" class="sm">256</text><path d="M60,20.0 L420,20.0" class="mute"/><text x="54" y="24.0" text-anchor="end" class="sm">512</text><rect x="63.0" y="20.0" width="30.0" height="170.0" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="78.0" y="204" text-anchor="middle" class="sm">1</text><rect x="99.0" y="38.9" width="30.0" height="151.1" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="114.0" y="204" text-anchor="middle" class="sm">2</text><rect x="135.0" y="57.8" width="30.0" height="132.2" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="150.0" y="204" text-anchor="middle" class="sm">3</text><rect x="171.0" y="76.7" width="30.0" height="113.3" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="186.0" y="204" text-anchor="middle" class="sm">4</text><rect x="207.0" y="95.6" width="30.0" height="94.4" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="222.0" y="204" text-anchor="middle" class="sm">5</text><rect x="243.0" y="114.4" width="30.0" height="75.6" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="258.0" y="204" text-anchor="middle" class="sm">6</text><rect x="279.0" y="133.3" width="30.0" height="56.7" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="294.0" y="204" text-anchor="middle" class="sm">7</text><rect x="315.0" y="152.2" width="30.0" height="37.8" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="330.0" y="204" text-anchor="middle" class="sm">8</text><rect x="351.0" y="171.1" width="30.0" height="18.9" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="366.0" y="204" text-anchor="middle" class="sm">9</text><rect x="387.0" y="190.0" width="30.0" height="0.0" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="402.0" y="204" text-anchor="middle" class="sm">10</text><path d="M60,63.9 L420,63.9" class="cb w2" stroke-dasharray="5 3"/><text x="240.0" y="220" text-anchor="middle" class="sm">zaman adımı</text><text x="440" y="34" class="sm">N = 1024: iş N − 1 = 1023</text><text x="440" y="54" class="sm">adım log₂N = 10</text><text x="440" y="74" class="sm">en yüksek paralellik 512</text><text x="440" y="94" class="sm">ortalama (N − 1)/log₂N = 102.3</text><text x="440" y="114" class="tb">(kesikli çizgi)</text><text x="440" y="134" class="sm"></text><text x="440" y="154" class="sm">Hızlanma (sınırsız kaynakla):</text><text x="440" y="174" class="sm">1023 / 10 = 102.3; 8 öğe için 8/3 = 2.67</text></svg><figcaption>Şekil 2 — İndirgeme ağacının paralelliği adım adım yarıya iner: ilk adımda 512, sonda 1; ortalama 102.3. Kaynak gereksinimindeki bu değişkenlik, ağacı donanıma eşlemeyi zorlaştırır (hesaplandı; dikey eksen logaritmik).</figcaption></figure>

### CUDA'da basit indirgeme {#basit}

Ağaç bütün thread'lerin işbirliğini ister. Bu bir grid boyunca mümkün değildir, çünkü bloklar arasında bariyer yoktur. İlk çekirdek bu yüzden **tek bir blokta** çalışır: $N$ öğe için $N/2$ thread'lik bir blok. Bir blok en çok 1024 thread taşıdığından en çok 2048 öğe işlenir; bu sınırı aşağıda kaldıracağız. İlk adımda bütün thread'ler ikişer öğe toplar; sonraki adımda yarısı düşer; son adımda tek bir thread toplamı üretir.

```cpp
unsigned int i = 2 * threadIdx.x;
for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) input[i] += input[i + stride];
    __syncthreads();
}
if (threadIdx.x == 0) *output = input[0];
```

Thread $t$, $2t$ konumunun **sahibidir**. $n$. yinelemede adım $2^n$'dir ve indeksi adımın katı olan thread'ler toplama yapar. `__syncthreads()`, bir yinelemenin bütün kısmi toplamları hedeflerine yazılmadan hiçbir thread'in sonraki yinelemeye başlamamasını sağlar. Örneğin ilk yinelemeden sonra çift konumlar çift toplamlarla değişmiş olmalıdır.

**Denetim ayrışması.** Bu çekirdek doğru sonucu verir, ama etkin ve etkin olmayan thread'leri yönetiş biçimi yüksek bir ayrışmaya yol açar. İkinci yinelemede bir warp'ın yalnız çift indeksli thread'leri toplama yapar: yürütme kaynaklarının yarısı boşa gider. Üçüncü yinelemede dörtte biri çalışır, beşincide 32'de biri.

256 öğede 128 thread, yani dört warp vardır. Dört warp da 1–5. yinelemelerde aynı ayrışma desenini gösterir. Altıncı yinelemede warp 1 ve 3 tamamen boşalır ve ayrışmaz (`% 64 == 0`). Warp 0 ve 2'de ise tek bir etkin thread kalır, kaynakların 31/32'si boşa gider. Yedinci yinelemede yalnız warp 0 çalışır, yine 31/32'si boşa. Sorunun kaynağı sahip konumların, dolayısıyla etkin thread'lerin, zamanla birbirinden giderek **uzaklaşmasıdır**.

**Bellek ayrışması.** İlk çekirdeğin bir sorunu daha vardır. Her yinelemede her thread iki global okuma (kendi konumu ve adım uzaklıktaki konum) ve bir yazma yapar. Komşu thread'lerin konumları komşu değildir; bir warp'ın konumları adım kadar aralıklıdır. İlk yinelemede iki öğe aralıklı okumalar iki istek tetikler ve dönen verinin yarısı kullanılmaz. İkincide dörtte biri kullanılır. Sonunda warp başına tek bir thread kalır ve okunan verinin 31/32'si atılır.

### Yakınsak eşleme {#yakinsak}

Çözüm, thread'leri ve sahip oldukları konumları zaman içinde **birbirine yakın** tutmaktır: adım zamanla artmamalı, azalmalıdır. Thread'ler dizinin ilk yarısına atanır. İlk yinelemede her thread dizinin yarısı kadar ötedeki öğeyi kendi konumuna ekler: thread 0 `input[8]`'i `input[0]`'a, thread 1 `input[9]`'u `input[1]`'e ekler. Her yinelemede etkin thread'lerin yarısı düşer ve kalanlar, etkin thread sayısı kadar ötedeki öğeyi ekler. 16 öğelik örnekte üçüncü yinelemede iki etkin thread kalır: thread 0 `input[2]`'yi `input[0]`'a, thread 1 `input[3]`'ü `input[1]`'e ekler.

```cpp
unsigned int i = threadIdx.x;
for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) input[i] += input[i + stride];
    __syncthreads();
}
```

Toplama yapan thread sayısı her yinelemede öncekiyle aynıdır. Fark, toplama yapan thread'lerin konumundadır: etkin thread'ler hep **bitişik indeksli**, yani aynı warp'lardadır.

256 öğelik örnekte ilk yinelemede bütün thread'ler etkindir ve ayrışma yoktur. İkinci yinelemede thread 0–63 toplar, 64–127 toplamaz: warp 0–1 tamamen etkin, warp 2–3 tamamen boştur. Her warp'ın bütün thread'leri aynı yolu izler, ayrışma yoktur. Ayrışma tamamen kalkmaz: dördüncü yinelemeden itibaren toplama yapan thread sayısı 32'nin altına düşer (16, 8, 4, 2, 1) ve son beş yinelemede tek bir warp ayrışır.

Bellek de iyileşir: komşu thread'ler her zaman komşu konumlara eriştiği için erişimler hep birleştirilir. Her okuma ya da yazmada warp başına tek bir istek yapılır. Yinelemeler ilerledikçe warp'lar bütünüyle düşer ve bu warp'lar hiç bellek erişimi yapmaz. Son beş yinelemede tek warp kaldığı için bant genişliği verimi düşer.

<figure class="fig wide"><svg viewBox="0 0 640 236" role="img" aria-label="Basit ve yakınsak indirgeme eşlemelerinde 16 öğelik bir dizide her yinelemenin yazılan ve okunan konumları"><text x="20" y="18" class="sm">basit: adım 1, 2, 4, 8</text><rect x="20" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="35" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="50" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="65" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="80" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="95" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="110" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="125" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="140" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="155" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="170" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="185" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="200" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="215" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="230" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="245" y="30" width="13" height="18" class="cb-f" opacity="1"/><text x="266" y="43" class="sm">8 thread</text><text x="16" y="43" text-anchor="end" class="sm"></text><rect x="20" y="64" width="13" height="18" class="ca-f" opacity="1"/><rect x="35" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="50" y="64" width="13" height="18" class="cb-f" opacity="1"/><rect x="65" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="80" y="64" width="13" height="18" class="ca-f" opacity="1"/><rect x="95" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="110" y="64" width="13" height="18" class="cb-f" opacity="1"/><rect x="125" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="140" y="64" width="13" height="18" class="ca-f" opacity="1"/><rect x="155" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="170" y="64" width="13" height="18" class="cb-f" opacity="1"/><rect x="185" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="200" y="64" width="13" height="18" class="ca-f" opacity="1"/><rect x="215" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="230" y="64" width="13" height="18" class="cb-f" opacity="1"/><rect x="245" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><text x="266" y="77" class="sm">4 thread</text><text x="16" y="77" text-anchor="end" class="sm"></text><rect x="20" y="98" width="13" height="18" class="ca-f" opacity="1"/><rect x="35" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="50" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="65" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="80" y="98" width="13" height="18" class="cb-f" opacity="1"/><rect x="95" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="110" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="125" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="140" y="98" width="13" height="18" class="ca-f" opacity="1"/><rect x="155" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="170" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="185" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="200" y="98" width="13" height="18" class="cb-f" opacity="1"/><rect x="215" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="230" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="245" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><text x="266" y="111" class="sm">2 thread</text><text x="16" y="111" text-anchor="end" class="sm"></text><rect x="20" y="132" width="13" height="18" class="ca-f" opacity="1"/><rect x="35" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="50" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="65" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="80" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="95" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="110" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="125" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="140" y="132" width="13" height="18" class="cb-f" opacity="1"/><rect x="155" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="170" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="185" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="200" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="215" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="230" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="245" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><text x="266" y="145" class="sm">1 thread</text><text x="16" y="145" text-anchor="end" class="sm"></text><text x="340" y="18" class="sm">yakınsak: adım 8, 4, 2, 1</text><rect x="340" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="355" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="370" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="385" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="400" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="415" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="430" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="445" y="30" width="13" height="18" class="ca-f" opacity="1"/><rect x="460" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="475" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="490" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="505" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="520" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="535" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="550" y="30" width="13" height="18" class="cb-f" opacity="1"/><rect x="565" y="30" width="13" height="18" class="cb-f" opacity="1"/><text x="586" y="43" class="sm">8 thread</text><text x="336" y="43" text-anchor="end" class="sm"></text><rect x="340" y="64" width="13" height="18" class="ca-f" opacity="1"/><rect x="355" y="64" width="13" height="18" class="ca-f" opacity="1"/><rect x="370" y="64" width="13" height="18" class="ca-f" opacity="1"/><rect x="385" y="64" width="13" height="18" class="ca-f" opacity="1"/><rect x="400" y="64" width="13" height="18" class="cb-f" opacity="1"/><rect x="415" y="64" width="13" height="18" class="cb-f" opacity="1"/><rect x="430" y="64" width="13" height="18" class="cb-f" opacity="1"/><rect x="445" y="64" width="13" height="18" class="cb-f" opacity="1"/><rect x="460" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="475" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="490" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="505" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="520" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="535" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="550" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="565" y="64" width="13" height="18" class="soft-a" opacity="0.5"/><text x="586" y="77" class="sm">4 thread</text><text x="336" y="77" text-anchor="end" class="sm"></text><rect x="340" y="98" width="13" height="18" class="ca-f" opacity="1"/><rect x="355" y="98" width="13" height="18" class="ca-f" opacity="1"/><rect x="370" y="98" width="13" height="18" class="cb-f" opacity="1"/><rect x="385" y="98" width="13" height="18" class="cb-f" opacity="1"/><rect x="400" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="415" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="430" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="445" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="460" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="475" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="490" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="505" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="520" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="535" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="550" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="565" y="98" width="13" height="18" class="soft-a" opacity="0.5"/><text x="586" y="111" class="sm">2 thread</text><text x="336" y="111" text-anchor="end" class="sm"></text><rect x="340" y="132" width="13" height="18" class="ca-f" opacity="1"/><rect x="355" y="132" width="13" height="18" class="cb-f" opacity="1"/><rect x="370" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="385" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="400" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="415" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="430" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="445" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="460" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="475" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="490" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="505" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="520" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="535" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="550" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><rect x="565" y="132" width="13" height="18" class="soft-a" opacity="0.5"/><text x="586" y="145" class="sm">1 thread</text><text x="336" y="145" text-anchor="end" class="sm"></text><rect x="20" y="172" width="12" height="12" class="ca-f"/><text x="36" y="182" class="sm">sahip konum (yazılır)</text><rect x="190" y="172" width="12" height="12" class="cb-f"/><text x="206" y="182" class="sm">ortak (okunur)</text><text x="20" y="206" class="sm">Basitte sahip konumlar 2, 4, 8 aralıklı: etkin thread'ler dağınık, her warp ayrışır, erişimler birleşmez.</text><text x="20" y="224" class="sm">Yakınsakta etkin thread'ler ve konumları hep baştadır: warp'lar bütünüyle boşalır, erişimler bitişiktir.</text></svg><figcaption>Şekil 1 — Aynı toplama sayısı, farklı yerleşim: basit eşlemede yazılan konumlar giderek uzaklaşır, yakınsak eşlemede bitişik kalır (sayılarak çizildi). Yakınsak eşleme, bitişik olmayan öğeleri birleştirdiği için işlecin değişmeli olmasını da ister.</figcaption></figure>

:::uyarı Kaynak düzeltmeleri (lect_08): yineleme sayısı ve değişme özelliği
**Yineleme sayısı.** Slayt 256 öğelik örnekten sonra ayrışan yineleme sayısının "ondan beşe" indiğini söylüyor. 256 öğede ($\log_2256=8$ yineleme) basit eşlemede ilk yineleme hariç 7 yineleme ayrışır, yakınsakta 5; yani **yediden beşe**. "Ondan beşe", 2048 öğelik (11 yineleme) bir blok için doğrudur. Aşağıdaki deney ikisini de sayıyor.

**Değişme özelliği.** Slayt işlecin birleşmeli ve değişmeli olmasını genel koşul olarak koyuyor. Basit eşleme yalnız **komşu** öğeleri, sıralarını koruyarak birleştirir: yalnız birleşmelilik gerekir. Yakınsak eşleme ise `input[0]`'a `input[8]`'i ekler; sonuç $x_0\oplus x_8\oplus\dots$ sırasındadır. Değişmeli olmayan bir işleçte, örneğin matris çarpımında ya da "ilk en büyüğün indeksi" gibi sıraya duyarlı bir işlemde, bu yanlış sonuç verir. Kayan nokta toplamında iki eşleme farklı sırada topladığı için son bitlerde farklı sonuç verir (aşağıda).
:::

Aşağıdaki deney iki eşlemeyi yineleme yineleme sayıyor. 256 öğede iş yapan warp sayısı yinelemeler boyunca toplam 27'den 12'ye iniyor. Global bellek sektörleri 381'den 102'ye düşüyor: yararlı bayt oranı %25'ten %94'e çıkıyor.

::lab l1

### Paylaşımlı bellek ve keyfi uzunluk {#paylasimli}

İkinci çekirdek her yinelemede kısmi toplamları global belleğe yazar ve aynı ya da başka thread'ler onları bir sonraki yinelemede yeniden okur. Paylaşımlı belleğin gecikmesi çok daha kısa, bant genişliği çok daha yüksektir. Kısmi toplamları paylaşımlı bellekte tutmak yürütmeyi hızlandırır. İlk adımda her thread iki global öğeyi okuyup toplar ve sonucu `input_s[t]`'ye yazar. Ağaç tamamen paylaşımlı bellekte döner, global belleğe yalnız sonuç yazılır. Bu sürüm girdiyi değiştirmediği için girdiyi koruma derdi de kalmaz.

**Keyfi uzunluk: hiyerarşik indirgeme.** Şimdiye kadarki çekirdekler tek blokla çalışır, çünkü `__syncthreads()` yalnız aynı bloğun thread'leri arasındadır. Bu, paralelliği blok başına en çok thread sayısıyla (1024) sınırlar. Milyonlarca, milyarlarca öğede daha fazla thread kullanmak isteriz. Bloklar arasında iyi bir bariyer olmadığı için blokların **bağımsız** çalışmasına izin verilir:

- Girdi, her biri bir bloğa uygun boyutta **bölümlere** ayrılır. Bölüm boyu $2\cdot\texttt{blockDim.x}$'tir ve her thread'in başlangıç konumu `segment = 2 * blockDim.x * blockIdx.x` ile belirlenir.
- Her blok kendi bölümünde ağacı bağımsız olarak çalıştırır.
- Ağaç bitince bölümün toplamı `input_s[0]`'dadır. Blok sonucunu thread 0 son çıktıya **atomik toplamayla** (`atomicAdd(output, input_s[0])`) ekler. Gridin bütün blokları bitince toplam `output`'tadır.

Seçenek, iki çekirdekli bir yaklaşımdır: birinci çekirdek blok toplamlarını bir diziye yazar, ikinci çekirdek o diziyi indirger. Atomik sürüm tek bir çekirdekle yetinir, ama kayan noktada sonuç blokların bitiş sırasına bağlıdır ve deterministik değildir.

### Thread kabalaştırma {#kabalastirma}

Şimdiye kadarki çekirdekler olabildiğince çok thread kullanıyor: $N$ öğe için $N/2$ thread. 1024 thread'lik bloklarla bu $N/2048$ blok demektir. Ama donanımın yürütme kaynakları sınırlıysa blokların yalnız bir kısmı paralel çalışabilir ve fazlası **sıraya** girer: biri bitince yenisi başlar. İndirgemeyi bloklara dağıtmanın bedeli yüksektir. Her blokta ağacın sonraki adımlarında warp'lar boşa düşer ve son warp ayrışır; bu az kullanım evresi başlatılan **her blokta** tekrarlanır. Bloklar gerçekten paralel çalışacaksa bu kaçınılmaz bir bedeldir. Ama donanım blokları zaten sıraya koyacaksa, bu işi kendimiz daha verimli yapmalıyız.

**Thread kabalaştırma** (coarsening), işin bir kısmını daha az thread'e sıralı olarak yükleyerek paralelleştirme ek yükünü azaltan eniyilemelerin genel adıdır. İndirgemede her bloğa daha çok öğe atanır. Her thread ağaca girmeden önce $2C$ öğeyi kendi yazmacında **sıralı** toplar. Bu evrede ağacın az kullanım sorunu yoktur: bütün thread'ler tam verimle çalışır. Sonra blok içi ağaç $2C$ kat daha az blokta, bir kez yürür.

Çarpanı büyütmenin bir sınırı vardır. Kabalaştırdıkça paralel yapılan iş azalır. Başlatılan blok sayısı donanımın aynı anda çalıştırabileceğinin altına düşerse paralel kaynaklardan tam yararlanılamaz. En iyi çarpan, donanımı doldurmaya yetecek kadar blok bırakandır; girdi boyuna ve aygıtın özelliklerine bağlıdır.

Aşağıdaki maliyet modeli (A100 benzeri, 108 SM) 2²⁴ öğede en iyi çarpanı C = 32 buluyor: 1024 blok, eşzamanlı kapasite 864. C = 1'de 32 768 blok başlatılıyor ve süreye en büyük katkıyı bloklar başına atomik işlemler ve sabit maliyetler yapıyor. C = 1024'te yalnız 32 blok kalıyor ve SM'lerin çoğu boş.

::lab l3

### Warp düzeyi işlemler {#warp}

Yakınsak çekirdeğin son beş yinelemesinin verimi hâlâ düşüktür: adım 32'nin altına inince son warp'taki boş thread'ler artar. Bunu iyileştirmenin yolu **warp düzeyi işlemlerdir**. Bir warp'ın thread'leri paylaşımlı bellek kullanmadan, doğrudan birbirlerinin yazmaçlarını okuyarak veri alışverişi yapar. Aşağıda "lane", thread'in warp içindeki indeksidir (0–31). Desteklenen veri taşıma desenleri:

- `__shfl_sync(mask, var, srcLane)`: belirtilen lane'den kopyalar (keyfi desen).
- `__shfl_up_sync(mask, var, delta)`: `delta` kadar alttaki lane'den kopyalar.
- `__shfl_down_sync(mask, var, delta)`: `delta` kadar üstteki lane'den kopyalar.
- `__shfl_xor_sync(mask, var, laneMask)`: lane kimliğinin `laneMask` ile XOR'u olan lane'den kopyalar (kelebek deseni).

Hem kaynak hem hedef thread işleme **katılmalıdır**. Bir koşul warp'ı dallara bölüyorsa shuffle yalnız kaynak ve hedef aynı (etkin) daldaysa çalışır; etkin olmayan bir hedeften okunan değer tanımsızdır. **Maske**, katılan ve yeniden birleşmesi gereken thread'leri belirtir; koşullu kodda buna göre ayarlanmalıdır. Diğer warp işlemleri şunlardır:

- `__ballot_sync(mask, pred)`: yüklemi doğru olan lane'lerin bitlerini 1 yapan bir maske döndürür.
- `__activemask()`: çağıran warp'ta o anda etkin olan thread'lerin maskesini verir; hangi thread'lerin aynı kodu yürüttüğünü gösterir.
- `__syncwarp()`: warp'ın thread'lerini eşzamanlar; `__syncthreads()`'in warp karşılığıdır.

Yararları şunlardır:

- paylaşımlı bellek kullanımını azaltmak ya da kaldırmak;
- iki ya da daha çok komutun yerine tek komut;
- daha az açık eşzamanlama;
- bir değeri tek komutla bütün warp'a yaymak;
- warp düzeyinde önek toplamı (M10.6);
- **atomik toplama**: bir warp'ın 32 atomik işlemini önce warp içinde birleştirip tek bir atomiğe indirmek.

**Shuffle ile indirgeme.** Ders çekirdeği üç aşamalıdır:

1. **Izgara adımlı döngü** (grid-stride loop): her thread `while (idx < N) { val += g[idx]; idx += gridDim.x * blockDim.x; }` ile girdinin bir bölümünü kendi `val` yazmacında toplar. Bu, kabalaştırmanın kendiliğinden hâlidir: grid boyutu girdiye değil donanıma göre seçilir.
2. **Warp içi ağaç:** `for (offset = 16; offset > 0; offset >>= 1) val += __shfl_down_sync(0xffffffff, val, offset);`. Beş adımda lane 0 warp'ın toplamını tutar. Paylaşımlı bellek de `__syncthreads()` da gerekmez.
3. **Warp toplamlarının birleştirilmesi:** her warp'ın lane 0'ı sonucunu `sdata[warpID]`'ye yazar (en çok 32 warp olduğu için 32 öğelik statik dizi yeter). Bir `__syncthreads()` gelir. Warp 0 bu 32 değeri aynı shuffle döngüsüyle indirger ve thread 0 sonucu `atomicAdd` ile çıktıya ekler.

:::uyarı Kaynak düzeltmesi (lect_08): warp işlemlerinin tarihi
Slayt warp düzeyi işlemlerin "Kepler (2014)" ile geldiğini, "CUDA 9 (2018)" ile iyileştirildiğini söylüyor. Kepler mimarisi ve `__shfl` komutları **2012**'de (GK104/GK110, hesap yeteneği 3.0) geldi. CUDA 9 Eylül **2017**'de yayımlandı. CUDA 9 eski `__shfl` işlemlerini maskeli `__shfl_*_sync` sürümleriyle değiştirdi; bu değişiklik, Volta'nın bağımsız thread zamanlamasında warp'ın kilitli adımda yürüdüğü varsayılamadığı için zorunluydu (M10.4).
:::

### Histogram: atomikler ve özelleştirme {#histogram}

Histogram bir indirgemedir, ama sonucu tek bir değer değil bir dizidir ve her öğenin hangi kutuya gideceği veriye bağlıdır. GPU'da en basit sürüm her thread'in kendi pikselini `atomicAdd(&hist[v], 1)` ile **global** histograma eklemesidir. Doğrudur, ama M10.3'teki gibi aynı kutuya yazan thread'ler sıraya girer. GPU'da çekişme CPU'dakinden de büyüktür: binlerce thread aynı 256 adrese yazar.

Çözüm yine **özelleştirmedir** (privatization). Her blok kendi histogramını **paylaşımlı bellekte** tutar. Blok içi atomikler paylaşımlı bellekte, global atomiklerden çok daha hızlıdır ve çekişme yalnız blok içindedir. Blok bitince kendi histogramını global histograma ekler: kutu başına bir atomik, yani blok başına 256 atomik. Thread kabalaştırma da burada işe yarar: her thread birçok piksel işler ve blok sayısı, dolayısıyla birleştirme maliyeti azalır.

Bir adım daha ileri, ardışık aynı değerleri **biriktirmektir** (aggregation). Bir thread art arda aynı kutuya düşen pikselleri sayıp tek bir atomikle ekler. Bu, bir bölgesi tek renkli görüntülerde çekişmeyi büyük ölçüde azaltır. Warp düzeyinde `__match_any_sync` ile aynı kutuya yazan lane'ler bulunup tek atomikte birleştirilebilir.

### Kayan nokta toplamı {#kayan}

Kayan nokta toplaması **birleşmeli değildir**: $(a+b)+c$ ile $a+(b+c)$ son bitlerde farklı olabilir. Bu yüzden paralel bir indirgemenin sonucu sıralı döngününkünden farklıdır. Hangisinin "doğru" olduğu sorusu yanıltıcıdır: ikisi de gerçek toplamın yaklaşıklarıdır. Asıl soru, hatanın $N$ ile nasıl büyüdüğüdür.

:::türetim Sıralı, ikili ve Kahan toplamının hatası
Her toplama işlemi göreli olarak en çok $\epsilon$ (fp32'de $2^{-24}\approx6\cdot10^{-8}$) hata ekler: $\mathrm{fl}(a+b)=(a+b)(1+\delta)$, $|\delta|\le\epsilon$.

**Sıralı toplam.** $s_k=\mathrm{fl}(s_{k-1}+x_k)$. İlk öğe $N-1$ toplamadan geçer, sonuncu bir toplamadan. Hatanın üst sınırı
$$|\hat s-s|\lesssim(N-1)\,\epsilon\sum|x_i|$$
olur. Pozitif veride kısmi toplam büyüdükçe küçük öğeler eklenirken giderek daha çok bitleri kaybolur. Uç bir örnek: $10^8$ tane 1.0'ı fp32'de sırayla toplamak $2^{24}=16\,777\,216$'da takılır. Bu noktada $16\,777\,216+1$ en yakın fp32 sayısına, yani yine $16\,777\,216$'ya yuvarlanır; sonraki 83 milyon toplama sonucu hiç değiştirmez.

**İkili (ağaç) toplam.** Her öğe yalnız $\lceil\log_2N\rceil$ toplamadan geçer; sınır $\lceil\log_2N\rceil\,\epsilon\sum|x_i|$'dir. Paralel indirgeme ağacı doğal olarak bu toplamdır, yani GPU indirgemesi genellikle sıralı CPU döngüsünden **daha doğrudur**.

**Kahan (telafili) toplamı.** Her adımda yuvarlamayla kaybolan kısım $c=(t-s)-y$ ile hesaplanır ve bir sonraki öğeden çıkarılır. Sınır $2\epsilon\sum|x_i|$'dir ve $N$'den bağımsızdır. Bedeli öğe başına dört kayan nokta işlemidir.

Gerçek hatalar rastgele işaretli biriktiği için sınırlardan çok küçüktür: sıralıda yaklaşık $\sqrt N\epsilon$, ikilide $\sqrt{\log N}\epsilon$.
:::

Aşağıdaki deney fp32'de üç yöntemi float64 başvurusuyla karşılaştırıyor. 10⁶ düzgün pozitif sayıda sıralı toplamın göreli hatası $2.5\cdot10^{-5}$, ikili toplamınki $4.4\cdot10^{-8}$, Kahan'ınki $1.9\cdot10^{-8}$. GPU tarzı "blok içinde sıralı, bloklar arasında ağaç" yöntemi ikili toplamla aynı düzeydedir.

::lab l2

### Özet {#ozet}

- İndirgeme için birleşmeli bir işleç ve birim eleman gerekir. Sıralı sürüm $N$ adımdır. Ağaçta iş $N-1$, adım $\log_2N$, ortalama paralellik $(N-1)/\log_2N$'dir (1024 öğede 102.3).
- Basit eşlemede (adım 1, 2, 4, …) sahip konumlar uzaklaşır: her warp ayrışır ve erişimler birleşmez. Yakınsak eşlemede (adım T, T/2, …, 1) etkin thread'ler bitişiktir. Ayrışan yineleme sayısı 256 öğede 7'den 5'e iner, erişimler birleşir; ama işlecin değişmeli olması gerekir.
- Paylaşımlı bellek kısmi toplamları yonga üstünde tutar. Hiyerarşik indirgemede bölümler bağımsız indirgenir ve sonuçlar `atomicAdd` ile birleştirilir.
- Thread kabalaştırmada her thread önce $2C$ öğeyi yazmaçta toplar. En iyi çarpan, donanımı dolduracak kadar blok bırakandır.
- Warp işlemleri (`__shfl_*_sync`, `__ballot_sync`, `__activemask`, `__syncwarp`) warp içi ağacı paylaşımlı bellek ve bariyer olmadan yapar.
- Histogram atomiklerle yapılır; blok başına paylaşımlı bellekte özelleştirme ve biriktirme çekişmeyi azaltır.
- Kayan nokta birleşmeli değildir. Hata sınırları sıralıda $O(N\epsilon)$, ikilide $O(\epsilon\log N)$, Kahan'da $O(\epsilon)$'dur; GPU ağacı sıralı döngüden genellikle daha doğrudur.
