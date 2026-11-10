### GPU ve CUDA {#giris}

M7.1 CUDA'nın yürütme modelini, birleştirmeyi ve karolamayı derin öğrenme çekirdekleri açısından tanıttı. Bu modül BİL 455'in dört CUDA dersini (lect_04–07) baştan sona işliyor: programlama modeli, bellek türleri, derleme ve uyumluluk, hizalama, birleştirme, bank çakışmaları, doluluk, hata denetimi, dal ayrışması ve döngü açma.

**GPU.** Grafik işlem birimi görsel görevler için tasarlandı: oyunlar ve bilgisayar destekli tasarımda 2-B/3-B grafik, bilgisayarla üretilmiş filmler ve özel efektler. Tüketici pazarının itişiyle GPU'lar çok yüksek kayan nokta verimine ve çok yüksek bellek bant genişliğine ulaştı. 2001 civarından beri programlanabilirler.

**CUDA** (Compute Unified Device Architecture), GPU'ları programlanabilir yapan donanım ve yazılımın adıdır. **GPGPU**, CUDA ya da OpenCL ile hesap yoğun problemleri GPU'da çözmektir. CUDA'nın C, C++ ve Fortran türleri vardır; CUDA C/C++, C'yi küçük bir anahtar sözcük kümesiyle genişletir. CUDA'dan önce de insanlar GPU'da kayan nokta hesabı yapıyordu: veriyi OpenGL veri yapılarına sarıp grafik işlemi taklit ederek. Grafik işlemlerinin çoğu kayan noktalı matris ve vektör işlemidir; GPU hâlâ grafik yaptığını "sanar", ama girdinin doğru düzenlenmesiyle yaptığı işlemler genel amaçlı bir hesaba karşılık gelir.

**Fark nereden geliyor?**

- **CPU:** başarımı çok karmaşık kontrol mantığına ve büyük, çok katmanlı önbelleklere dayanır. Bu, rastgele erişimli, bağımsız görev paralel işler için mükemmel bir birleşimdir.
- **GPU:** başarımı veri düzeyinde paralelliğe dayanır. Kontrol mantığı daha basit, önbellek daha küçüktür ve ikisi de daha az silisyum alır. GPU transistörlerinin daha büyük kısmını hesaba ayırır. Bellek türleri de daha hızlıdır (daha yüksek saat) ve özel veri yerleşimleriyle eniyilenmiştir.

Bunun sonucu, GPU'nun tek komut çok thread'li (SIMD/SIMT) yürütmeye çok daha iyi uymasıdır.

**SIMD ve SIMT.**

- **SIMD:** aynı komut birçok veri öğesinde paralel yürütülür.
- **SIMT:** aynı komutu birçok **thread** paralel yürütür.

SIMT, SIMD'nin daha esnek bir biçimidir: farklı thread'lere farklı görevler verilebilir, ama thread'ler farklı yollara saptıkça (ayrışma) başarım düşer. Akış denetimi basitleştiği için silisyum hesaba kalır. Aritmetik yoğunluk yüksekse (bellek erişimi başına çok komut), bellek gecikmesi büyük önbelleklerle değil, hesapla gizlenir. Veri paralel programlamanın özü öğeleri thread'lere eşlemektir. 3-B görüntüleme, görüntü ve video işleme, kodlama, stereo görüş, nesne tanıma, fizik simülasyonu, hesaplamalı finans ve biyoloji bu modele uyar.

**Neden çok çekirdek?** Moore yasası sürüyor ama hızı azalıyor. Frekans artırmak artık mümkün değildir: güç kabaca frekansın küpüyle büyür (M10.1), yüksek frekanslı saati yonga boyunca üretmek ve dağıtmak da zordur. Çözüm çok çekirdektir. Düşük frekanslı çekirdekler doğal olarak güç verimlidir, ama paralel yazılım ister. Bazı görevler doğası gereği sıralıdır ve CPU'ya uyar. Her işe en iyi tek bir işlemci yoktur; platformlar bu yüzden **heterojendir** (OpenCL bu noktada devreye girer). Gömülü ve mobil sistemlerde güç tüketimi sınırları da çok çekirdeği zorunlu kılar.

**CPU–GPU yerleşimleri.** Ayrık GPU'lar PCIe üzerinden bağlanır. AMD Athlon 64 ve Intel Nehalem, bellek denetleyicisini CPU kalıbına alarak ana bellek gecikmesini düşürdü. AMD'nin APU'ları CPU ile GPU'yu tek yongada birleştirir. hUMA, iki işlemcinin bellek uzaylarını birleştirip aralarında önbellek tutarlılığını da sağlar (M10.1).

**CUDA'nın genel yaklaşımı:**

1. Kodu çözümle, başarım darboğazlarını bul: çalışma süresinin büyük kısmını alan parçaları profil araçlarıyla belirle. Sabit disk erişimi, kullanıcı girdisi ya da ağ akışı beklemek bu tür adaylar değildir.
2. Veri paralel kod parçalarını bul. Bazı algoritmalar doğası gereği sıralıdır: Newton yöntemi, sıkıştırma algoritmalarının çoğu.
3. Gereken girdiyi CPU belleğinden GPU belleğine (PCIe üzerinden) kopyala. İstisnası sıfır kopyalı yürütmedir: yeni mimarilerde CPU ve GPU aynı DRAM'i paylaşabilir.
4. GPU'da hesapla: iyi bir thread–veri eşlemesi tasarla ve çekirdekleri yaz.
5. Sonuçları CPU'ya geri kopyala.
6. GPU sonuçlarının CPU sonuçlarıyla makul bir tolerans içinde eşleştiğini denetle. Kayan nokta işlemleri değişmelidir ($a+b=b+a$) ama **birleşmeli değildir** ($(a+b)+c\ne a+(b+c)$); paralel indirgemeler toplamı farklı sırada yapar.
7. Yinele: elde edilen hızlanma GPU'nun kuramsal tepe GFLOPS ve GB/s değerlerine ne kadar yakın? Değilse çekirdekleri eniyile.

Kazanç şu durumlarda beklenir:

- aynı işlem bütün veri noktalarına uygulanıyorsa;
- çoğunlukla tek duyarlıklı (float) hesap yapılıyorsa (çift duyarlık desteklenir ama özellikle eski ve tüketici mimarilerinde yavaştır);
- GPU'ya taşınan veri, veri başına hesaba göre küçükse.

### Programlama modeli {#model}

**Terimler.**

- **Ana makine ve aygıt:** GPU bir **yardımcı işlemcidir**. CPU ana makinedir (host): aygıtı yürütmeye hazırlar ve iş gönderir. GPU aygıttır (device): gönderilen işi yürütür ve sonucu döndürür.
- **Akış çok işlemcisi** (SM): GPU'nun yapı taşıdır ve kabaca bir CPU çekirdeğine benzer. Her SM skaler işlemcilerin (SP), yerel önbelleğin ve özel fonksiyon birimlerinin birleşimidir.
- **Thread'ler:** iş öğeleridir. Açıkça yaratılıp zamanlanmaları gerekmez; yaratma ve zamanlama GPU donanımınca yönetilir.
- **Warp:** birlikte yaratılan, zamanlanan ve yürütülen 32 thread.
- **Thread bloğu** (CTA): aynı SM'de çalışan thread grubu.
- **Çekirdek** (kernel, `__global__`): GPU fonksiyonu.

**Fonksiyon nitelikleri:**

| nitelik | nerede çalışır | nereden çağrılır |
|---|---|---|
| `__global__` | aygıt | ana makine (çekirdek) |
| `__device__` | aygıt | aygıt |
| `__host__` | ana makine | ana makine (varsayılan) |

Bir çekirdek `kernel<<<blokSayısı, blokBaşınaThread>>>(...)` ile başlatılır. Her thread kendisini `threadIdx` (bloktaki konumu, 3 bileşenli) ve `blockIdx` (griddeki bloğun konumu) ile tanır; `blockDim` ve `gridDim` boyutları verir. Pratik kural: **thread boyutluluğu = veri boyutluluğu.** Vektör için 1-B, matris için 2-B, hacim için 3-B kullanılır. Bir bloktaki thread'in doğrusal indeksi
$$\text{tid}=\texttt{threadIdx.z}\cdot B_xB_y+\texttt{threadIdx.y}\cdot B_x+\texttt{threadIdx.x}$$
olur ($B_x=\texttt{blockDim.x}$). Warp'lar bu indeksin ardışık 32'lik gruplarıdır: 0–31 birinci warp, 32–63 ikinci warp. Eşleme deterministiktir. Griddeki genel indeks 1-B'de `blockIdx.x * blockDim.x + threadIdx.x`'tir. Thread grubu bir veri dizisinden uzunsa sınır denetimi (`if (i < N)`) gerekir.

**Sınırlar.** Bugünkü GPU'larda bir blok en çok 1024 thread içerir; hesap yeteneği 1.x'te 512'ydi. Blok koordinatı 1.x'te 2-B, 2.0'dan beri 3-B'dir. Dersin ilk matris çarpımı örneği tek bir blokla çalışır ve yalnız `threadIdx` kullanır. Bu yüzden ancak bir bloğa sığan matrisleri işler: 1.x'in 512 sınırıyla 16 × 16. Gerçek çözüm 2-B bir blok gridi ve `row = blockIdx.y * TILE + threadIdx.y`, `col = blockIdx.x * TILE + threadIdx.x` indeksleridir.

**Bloklar bağımsızdır.** Bloklar herhangi bir sırada, paralel ya da seri yürütülebilmelidir. Hangi bloğun hangi SM'de çalışacağını programcı denetleyemez. Küçük indeksli bir bloğun önce başlayacağı da varsayılamaz. Bu bağımsızlık, aynı programın 2 SM'li bir GPU'da da 132 SM'li bir GPU'da da çalışmasını, yani **ölçeklenebilirliği** sağlar: çalışma zamanı bekleyen blokların bir listesini tutar ve SM'ler boşaldıkça yenilerini atar. CUDA yaklaşımı iki düzeylidir. Problem önce bağımsız çözülebilen kaba alt problemlere (bloklara) bölünür; bloklar arasında etkileşim yoktur. Her alt problem de blok içindeki thread'lerin **işbirliğiyle** çözülen ince parçalara bölünür; yerel etkileşimler blok içinde kalır.

**Blok içi işbirliği.** Bir bloğun thread'leri paylaşımlı bellekle veri paylaşır ve `__syncthreads()` bariyeriyle eşgüdüm sağlar: bloktaki bütün thread'ler bariyere ulaşmadan hiçbiri devam etmez. Örneğin bir karo paylaşımlı belleğe yüklenir, sonra kullanılır. Bariyer **bloktaki bütün thread'lerce** yürütülmelidir. Koşullu kodda yalnız bazı thread'lerin ulaştığı bir `__syncthreads()` tanımsız davranıştır; tek istisna, koşulun bloktaki her thread için aynı sonucu vermesidir.

Blok içinde eşzamanlama mümkün olduğu için çalışma zamanı bir bloğun bütün thread'lerini aynı SM'ye birlikte atar. Bloklar arası eşzamanlama ise uzun süre olanaksızdı: geçici çözümler `__threadfence()` gibi bellek çitleriyle yapılıyordu. CUDA 9'daki **işbirlikçi gruplar** (cooperative groups), işbirlikçi başlatmayla bütün grid için eşzamanlamayı Pascal ve sonrasında mümkün kıldı.

:::uyarı Kaynak düzeltmesi (lect_05 slayt 44, lect_07): __syncthreads ve program sayacı
Slayt 44 `__syncthreads()`'in "thread'leri değil warp'ları eşzamanladığını" söylüyor. lect_07 de bir warp'ın bütün thread'lerinin "bir program sayacını paylaştığını" söylüyor. Bu, Volta (2017) öncesi mimariler için doğruydu. Volta'dan beri **bağımsız thread zamanlaması** vardır: her thread'in kendi program sayacı ve çağrı yığını vardır. Bir warp'ın thread'leri ayrışık yollarda iç içe ilerleyebilir ve yeniden birleşmeleri garanti değildir. `__syncthreads()` bloktaki **thread'leri** eşzamanlar. Warp içi işbirliği yapan kod (shuffle, oy fonksiyonları) açık bir maske ister (`__shfl_sync(mask, …)`, `__syncwarp()`). "Warp zaten kilitli adımda yürür" varsayımıyla yazılmış eski kod Volta ve sonrasında yanlış sonuç verebilir. Ayrışan yolların sırayla yürütülmesi ve bunun başarım maliyeti ise sürüyor.
:::

**Thread blok kümeleri.** Hesap yeteneği 9.0 (Hopper) ile isteğe bağlı bir hiyerarşi düzeyi geldi: **thread blok kümeleri**. Bir bloğun thread'lerinin bir SM'de birlikte zamanlanması gibi, bir kümenin blokları da bir **GPU işlem kümesinde** (GPC) birlikte zamanlanır. Ayrıntılar şöyledir:

- **Boyut ve biçim:** kümeler 1, 2 ya da 3 boyutlu olabilir. Taşınabilir en büyük küme boyu 8 bloktur; daha küçük yapılandırmalarda (MIG) azalır.
- **Tanımlama:** küme boyutu derleme zamanında `__cluster_dims__(X, Y, Z)` niteliğiyle ya da çalışma zamanında `cudaLaunchKernelEx` ile verilir. Grid boyutu küme boyutunun katı olmalıdır.
- **Eşzamanlama ve sorgu:** kümedeki bloklar `cluster.sync()` ile donanım destekli eşzamanlanır. `num_threads()` ve `num_blocks()` boyutu verir.
- **Dağıtık paylaşımlı bellek:** bloklar birbirlerinin paylaşımlı belleğini okuyup yazabilir ve üzerinde atomik işlem yapabilir.

Hesap hiyerarşisinin bir örneği Ampere'dir (GA100): iki SM bir doku işlem kümesi (TPC), sekiz TPC (16 SM) bir GPC, sekiz GPC tam GPU'yu oluşturur.

:::uyarı Kaynak düzeltmeleri (lect_04 slayt 5, 51, 54, 66; lect_06 slayt 34)
- **Hesap yetenekleri:** Fermi'nin hesap yetenekleri 2.0 ve 2.1'dir; compute_30 Kepler'e aittir. Ana numaralar 1 (Tesla), 2 (Fermi), 3 (Kepler), 5 (Maxwell; 4 yoktur), 6 (Pascal), 7 (Volta 7.0/7.2, Turing 7.5), 8 (Ampere 8.0/8.6/8.7, Ada 8.9), 9 (Hopper) ve 10/12'dir (Blackwell).
- **Thread blok kümeleri:** slayt 51 bu kümelerin "CUDA 9'da" geldiğini söylüyor. Kümeler hesap yeteneği 9.0 (Hopper) ve CUDA 12 ile geldi; CUDA 9'un getirdiği işbirlikçi gruplardır.
- **MIG:** slayt 54'teki "sekiz MIG dilimi" de doğru değil. A100 en çok **7** MIG örneğine bölünebilir. Üretilen A100'lerde 8 GPC'den 7'si etkindir (108 SM); her MIG örneği bir GPC dilimi alır.
:::

### Bellek ve thread hiyerarşisi {#bellek}

| bellek | kapsam | ömür | konum | not |
|---|---|---|---|---|
| yazmaç | thread | thread | yonga üstü | en hızlı; SM başına sabit sayıda |
| yerel | thread | thread | aygıt belleği (DRAM) | yazmaç taşması, dizinlenemeyen diziler |
| paylaşımlı | blok | blok | yonga üstü | kullanıcının yönettiği önbellek, bank'lı |
| global | grid ve ana makine | uygulama | DRAM | büyük, yavaş (yüzlerce çevrim) |
| sabit | grid ve ana makine | uygulama | DRAM + sabit önbelleği | salt okunur, 64 KB, yayın |
| doku | grid ve ana makine | uygulama | DRAM + doku önbelleği | salt okunur, 2-B yerellik, süzme |

**Değişken türleri:**

- **Otomatik skaler değişkenler** yazmaçlara konur: her thread'in özel kopyası vardır ve thread bitince yok olur.
- **`__shared__`** değişkenler blok kapsamlıdır: bloktaki bütün thread'ler aynı kopyayı görür, çekirdek bitince içerik kaybolur.
- **`__constant__`** değişkenler fonksiyon gövdesi dışında bildirilir, bütün gridlerce görülür ve uygulama boyunca yaşar. Ana makine `cudaMemcpyToSymbol` ile doldurur.

Yazmaçlar hızlıdır, ama SM başına toplam sayı sabittir ve SM'deki thread'lere bölünür. Derlenen kod thread başına sınırdan çok yazmaç isterse fazlası **yerel belleğe taşar** (spill). Yerel bellek aslında thread'e özel global bellektir; çok yavaştır. `--ptxas-options=-v` derleyici seçeneği çekirdeğin yazmaç ve yerel bellek kullanımını gösterir. Derleyicinin yerel belleğe koyacağı değişkenler şunlardır:

- sabit indekslerle dizinlendiği anlaşılamayan diziler;
- yazmaçlara sığmayan büyük yapılar;
- çekirdek yazmaç sınırını aşınca herhangi bir değişken.

Yerel bellek ardışık 32 bitlik sözcükleri ardışık thread'lere dağıtacak biçimde düzenlenir. Bu yüzden bir warp'ın bütün thread'leri aynı göreli adrese erişirse erişimler tam birleştirilir.

**Sabit bellek.** Global bellekte durur ama sabit önbelleğinde tutulur. Bir warp'ın isteği, warp'taki farklı adres sayısı kadar ayrı isteğe bölünür. En iyi durum bütün thread'lerin aynı adresi okumasıdır: tek istek, bütün warp'a **yayın**. Bir evrişim filtresinin katsayıları tipik örnektir (M10.7). Aynı warp'taki thread'ler farklı adresler okursa sabit bellek global bellekten de yavaş olabilir.

**Doku belleği.** Grafik donanımının bir alt kümesidir: salt okunurdur (yazılabilir türü yüzey, surface belleğidir). Doku önbelleği **2-B uzamsal yerellik** için eniyilenmiştir: aynı warp'ın 2-B'de yakın adresleri okuyan thread'leri en iyi başarımı alır. Ek olanakları da vardır:

- adres hesapları çekirdeğin dışında, özel birimlerde yapılır;
- 8 ve 16 bitlik tamsayılar isteğe bağlı olarak $[0,1]$ ya da $[-1,1]$ aralığında normalize kayan noktaya çevrilir;
- **ücretsiz doğrusal süzme** vardır: 1-B'de doğrusal, 2-B'de bilineer, 3-B'de üç doğrusal enterpolasyon;
- **sınır dışı adresleme kipleri** donanımda çözülür: sıkıştır (clamp, kenar pikselini tekrarla), sınır (border, sıfır), sar (wrap), yansıt (mirror).

Dokular **doku nesnesiyle** okunur. Doku nesnesi `cudaCreateTextureObject` ile bir kaynak tanımından (doğrusal bellek ya da opak bir **CUDA dizisi**) ve bir doku tanımından (koordinat normalizasyonu, adresleme ve süzme kipi, okuma kipi) kurulur. Okuma, doğrusal bellekte `tex1Dfetch`, CUDA dizilerinde `tex1D/2D/3D` ile yapılır. Sınırlamalar da vardır: çekirdek içinden yazılamaz, ve doğrusal bellekten bir CUDA dizisine kopyalama zaman alır, dolayısıyla sonraki hesap buna değmelidir. Görüntü işlemede (enterpolasyon, yeniden örnekleme, kenar koşulları) dokular büyük kolaylıktır.

**L2'de kalıcı erişim.** Hesap yeteneği 8.0 ve CUDA 11 ile L2 önbelleğinin bir kısmı, tekrar tekrar erişilen ("kalıcı") veriye ayrılabilir. Yalnız bir kez erişilen ("akış") veri bu ayrılmış kısmı kirletmez.

### Bellek erişim verimi: CGMA {#cgma}

Dersin ilk matris çarpımı çekirdeği her thread'e bir çıktı elemanı verir. İç döngünün her turunda iki global bellek erişimi (M ve N'den birer eleman) bir çarpma ve bir toplama için yapılır. **CGMA** (compute to global memory access), bir program bölgesinde global bellek erişimi başına yapılan kayan nokta işlemi sayısıdır; burada 1'dir. G80'in global bellek bant genişliği 86.4 GB/s'dir. 4 baytlık float'larla bu saniyede en çok 21.6 milyar veri, yani CGMA 1 ile en çok 21.6 GFLOPS eder. G80'in tepe başarımı 367 GFLOPS'tur: çekirdek tepenin %5.9'unda kalır.

<figure class="fig wide"><svg viewBox="0 0 640 246" role="img" aria-label="CGMA oranına göre ulaşılabilir başarım: G80 ve A100 için bellek ve hesap tavanları"><path d="M60,210.0 L390,210.0" class="mute"/><text x="54" y="214.0" text-anchor="end" class="sm">1</text><path d="M60,172.0 L390,172.0" class="mute"/><text x="54" y="176.0" text-anchor="end" class="sm">10</text><path d="M60,134.0 L390,134.0" class="mute"/><text x="54" y="138.0" text-anchor="end" class="sm">100</text><path d="M60,96.0 L390,96.0" class="mute"/><text x="54" y="100.0" text-anchor="end" class="sm">1k</text><path d="M60,58.0 L390,58.0" class="mute"/><text x="54" y="62.0" text-anchor="end" class="sm">10k</text><path d="M60,20.0 L390,20.0" class="mute"/><text x="54" y="24.0" text-anchor="end" class="sm">100k</text><text x="107.1" y="226" text-anchor="middle" class="sm">1</text><text x="201.4" y="226" text-anchor="middle" class="sm">10</text><text x="295.7" y="226" text-anchor="middle" class="sm">100</text><text x="390.0" y="226" text-anchor="middle" class="sm">1000</text><text x="225.0" y="242" text-anchor="middle" class="sm">CGMA (FLOP / 4 baytlık global erişim)</text><path d="M60.0,178.3 L61.7,177.6 L63.3,177.0 L65.0,176.3 L66.6,175.6 L68.3,174.9 L69.9,174.3 L71.6,173.6 L73.3,172.9 L74.9,172.3 L76.6,171.6 L78.2,170.9 L79.9,170.3 L81.6,169.6 L83.2,168.9 L84.9,168.3 L86.5,167.6 L88.2,166.9 L89.8,166.3 L91.5,165.6 L93.2,164.9 L94.8,164.3 L96.5,163.6 L98.1,162.9 L99.8,162.3 L101.5,161.6 L103.1,160.9 L104.8,160.2 L106.4,159.6 L108.1,158.9 L109.7,158.2 L111.4,157.6 L113.1,156.9 L114.7,156.2 L116.4,155.6 L118.0,154.9 L119.7,154.2 L121.4,153.6 L123.0,152.9 L124.7,152.2 L126.3,151.6 L128.0,150.9 L129.6,150.2 L131.3,149.6 L133.0,148.9 L134.6,148.2 L136.3,147.5 L137.9,146.9 L139.6,146.2 L141.3,145.5 L142.9,144.9 L144.6,144.2 L146.2,143.5 L147.9,142.9 L149.5,142.2 L151.2,141.5 L152.9,140.9 L154.5,140.2 L156.2,139.5 L157.8,138.9 L159.5,138.2 L161.2,137.5 L162.8,136.9 L164.5,136.2 L166.1,135.5 L167.8,134.8 L169.4,134.2 L171.1,133.5 L172.8,132.8 L174.4,132.2 L176.1,131.5 L177.7,130.8 L179.4,130.2 L181.1,129.5 L182.7,128.8 L184.4,128.2 L186.0,127.5 L187.7,126.8 L189.3,126.2 L191.0,125.5 L192.7,124.8 L194.3,124.2 L196.0,123.5 L197.6,122.8 L199.3,122.2 L201.0,121.5 L202.6,120.8 L204.3,120.1 L205.9,119.5 L207.6,118.8 L209.2,118.1 L210.9,117.5 L212.6,116.8 L214.2,116.1 L215.9,115.5 L217.5,114.8 L219.2,114.1 L220.9,113.5 L222.5,112.8 L224.2,112.5 L225.8,112.5 L227.5,112.5 L229.1,112.5 L230.8,112.5 L232.5,112.5 L234.1,112.5 L235.8,112.5 L237.4,112.5 L239.1,112.5 L240.8,112.5 L242.4,112.5 L244.1,112.5 L245.7,112.5 L247.4,112.5 L249.0,112.5 L250.7,112.5 L252.4,112.5 L254.0,112.5 L255.7,112.5 L257.3,112.5 L259.0,112.5 L260.7,112.5 L262.3,112.5 L264.0,112.5 L265.6,112.5 L267.3,112.5 L268.9,112.5 L270.6,112.5 L272.3,112.5 L273.9,112.5 L275.6,112.5 L277.2,112.5 L278.9,112.5 L280.6,112.5 L282.2,112.5 L283.9,112.5 L285.5,112.5 L287.2,112.5 L288.8,112.5 L290.5,112.5 L292.2,112.5 L293.8,112.5 L295.5,112.5 L297.1,112.5 L298.8,112.5 L300.5,112.5 L302.1,112.5 L303.8,112.5 L305.4,112.5 L307.1,112.5 L308.7,112.5 L310.4,112.5 L312.1,112.5 L313.7,112.5 L315.4,112.5 L317.0,112.5 L318.7,112.5 L320.4,112.5 L322.0,112.5 L323.7,112.5 L325.3,112.5 L327.0,112.5 L328.6,112.5 L330.3,112.5 L332.0,112.5 L333.6,112.5 L335.3,112.5 L336.9,112.5 L338.6,112.5 L340.3,112.5 L341.9,112.5 L343.6,112.5 L345.2,112.5 L346.9,112.5 L348.5,112.5 L350.2,112.5 L351.9,112.5 L353.5,112.5 L355.2,112.5 L356.8,112.5 L358.5,112.5 L360.2,112.5 L361.8,112.5 L363.5,112.5 L365.1,112.5 L366.8,112.5 L368.4,112.5 L370.1,112.5 L371.8,112.5 L373.4,112.5 L375.1,112.5 L376.7,112.5 L378.4,112.5 L380.1,112.5 L381.7,112.5 L383.4,112.5 L385.0,112.5 L386.7,112.5 L388.3,112.5 L390.0,112.5" class="cb w2" fill="none"/><circle cx="223.1" cy="112.5" r="4" class="cb-f"/><path d="M60.0,130.6 L61.7,129.9 L63.3,129.3 L65.0,128.6 L66.6,127.9 L68.3,127.3 L69.9,126.6 L71.6,125.9 L73.3,125.2 L74.9,124.6 L76.6,123.9 L78.2,123.2 L79.9,122.6 L81.6,121.9 L83.2,121.2 L84.9,120.6 L86.5,119.9 L88.2,119.2 L89.8,118.6 L91.5,117.9 L93.2,117.2 L94.8,116.6 L96.5,115.9 L98.1,115.2 L99.8,114.6 L101.5,113.9 L103.1,113.2 L104.8,112.5 L106.4,111.9 L108.1,111.2 L109.7,110.5 L111.4,109.9 L113.1,109.2 L114.7,108.5 L116.4,107.9 L118.0,107.2 L119.7,106.5 L121.4,105.9 L123.0,105.2 L124.7,104.5 L126.3,103.9 L128.0,103.2 L129.6,102.5 L131.3,101.9 L133.0,101.2 L134.6,100.5 L136.3,99.8 L137.9,99.2 L139.6,98.5 L141.3,97.8 L142.9,97.2 L144.6,96.5 L146.2,95.8 L147.9,95.2 L149.5,94.5 L151.2,93.8 L152.9,93.2 L154.5,92.5 L156.2,91.8 L157.8,91.2 L159.5,90.5 L161.2,89.8 L162.8,89.2 L164.5,88.5 L166.1,87.8 L167.8,87.2 L169.4,86.5 L171.1,85.8 L172.8,85.1 L174.4,84.5 L176.1,83.8 L177.7,83.1 L179.4,82.5 L181.1,81.8 L182.7,81.1 L184.4,80.5 L186.0,79.8 L187.7,79.1 L189.3,78.5 L191.0,77.8 L192.7,77.1 L194.3,76.5 L196.0,75.8 L197.6,75.1 L199.3,74.5 L201.0,73.8 L202.6,73.1 L204.3,72.4 L205.9,71.8 L207.6,71.1 L209.2,70.4 L210.9,69.8 L212.6,69.1 L214.2,68.4 L215.9,67.8 L217.5,67.1 L219.2,66.4 L220.9,65.8 L222.5,65.1 L224.2,64.4 L225.8,63.8 L227.5,63.1 L229.1,62.4 L230.8,61.8 L232.5,61.1 L234.1,60.4 L235.8,59.7 L237.4,59.1 L239.1,58.4 L240.8,57.7 L242.4,57.1 L244.1,56.4 L245.7,55.7 L247.4,55.1 L249.0,54.4 L250.7,53.7 L252.4,53.1 L254.0,52.4 L255.7,51.7 L257.3,51.1 L259.0,50.4 L260.7,49.7 L262.3,49.1 L264.0,48.4 L265.6,47.7 L267.3,47.0 L268.9,47.0 L270.6,47.0 L272.3,47.0 L273.9,47.0 L275.6,47.0 L277.2,47.0 L278.9,47.0 L280.6,47.0 L282.2,47.0 L283.9,47.0 L285.5,47.0 L287.2,47.0 L288.8,47.0 L290.5,47.0 L292.2,47.0 L293.8,47.0 L295.5,47.0 L297.1,47.0 L298.8,47.0 L300.5,47.0 L302.1,47.0 L303.8,47.0 L305.4,47.0 L307.1,47.0 L308.7,47.0 L310.4,47.0 L312.1,47.0 L313.7,47.0 L315.4,47.0 L317.0,47.0 L318.7,47.0 L320.4,47.0 L322.0,47.0 L323.7,47.0 L325.3,47.0 L327.0,47.0 L328.6,47.0 L330.3,47.0 L332.0,47.0 L333.6,47.0 L335.3,47.0 L336.9,47.0 L338.6,47.0 L340.3,47.0 L341.9,47.0 L343.6,47.0 L345.2,47.0 L346.9,47.0 L348.5,47.0 L350.2,47.0 L351.9,47.0 L353.5,47.0 L355.2,47.0 L356.8,47.0 L358.5,47.0 L360.2,47.0 L361.8,47.0 L363.5,47.0 L365.1,47.0 L366.8,47.0 L368.4,47.0 L370.1,47.0 L371.8,47.0 L373.4,47.0 L375.1,47.0 L376.7,47.0 L378.4,47.0 L380.1,47.0 L381.7,47.0 L383.4,47.0 L385.0,47.0 L386.7,47.0 L388.3,47.0 L390.0,47.0" class="ca w2" fill="none"/><circle cx="267.5" cy="47.0" r="4" class="ca-f"/><text x="405" y="34" class="tb">G80: 86.4 GB/s, 367 GFLOPS</text><text x="405" y="52" class="sm">CGMA 1 → 21.6 GFLOPS (5.9%)</text><text x="405" y="70" class="sm">karo 16 → CGMA 16 → 346 GFLOPS</text><text x="405" y="88" class="sm">tavana geçiş: CGMA 17.0</text><text x="405" y="120" class="ta">A100: 1555 GB/s, 19.5 TFLOPS</text><text x="405" y="138" class="sm">CGMA 1 → 389 GFLOPS (2.0%)</text><text x="405" y="156" class="sm">tavana geçiş: CGMA 50</text><text x="405" y="188" class="sm">Hesap, belleğe göre daha hızlı</text><text x="405" y="204" class="sm">büyüdüğü için gereken CGMA artıyor.</text></svg><figcaption>Şekil 2 — Ulaşılabilir başarım = min(tepe hesap, bant genişliği × CGMA / 4 bayt). Naif matmul (CGMA 1) G80'de tepenin %5.9'unda kalır; 16 × 16 karo tavana yaklaştırır. Yeni GPU'larda hesap bant genişliğinden hızlı büyüdüğü için gereken yeniden kullanım daha yüksektir (hesaplandı; eksenler logaritmik).</figcaption></figure>

**Karolama.** 2 × 2'lik bloklarla küçük bir örnek, aynı bloktaki thread'lerin M ve N'nin aynı elemanlarını tekrar tekrar okuduğunu gösteriyor. Örneğin thread (0, 0) ve (1, 0) M'nin 0. satırının bütün elemanlarını okur. Thread'ler işbirliği yapıp her elemanı global bellekten bir kez yüklerse trafik yarıya iner. Genel olarak $T\times T$'lik bloklarla global trafik $T$ kat azalır. 16 × 16 bloklarla, paylaşımlı bellek olmadan, $16\cdot16\cdot2W$ okuma yapılır (her thread $2W$ eleman okur); paylaşımlı bellekle $16\cdot2W$. Oran 16'dır. Karolu çekirdek M ve N'nin $T\times T$'lik karolarını paylaşımlı belleğe işbirliğiyle yükler (`Mds[ty][tx] = Md[...]`), `__syncthreads()` ile bekler, karodaki $T$ çarpımı yapar ve yeniden `__syncthreads()` çağırır. İki bariyerin görevleri farklıdır:

- **İlki** (yüklemeden sonra), bütün karo yüklenmeden hiçbir thread'in onu okumamasını sağlar: okuma-yazma sırası, yani gerçek bağımlılık.
- **İkincisi** (hesaptan sonra), bütün thread'ler bu karoyla işini bitirmeden kimsenin bir sonraki karoyu üzerine yazmamasını sağlar: yazma-okuma sonrası (WAR) tehlikesi.

$T\times T$ karolarla CGMA $T$ olur (M7.1, M10.7). Dikdörtgen bloklarda (örneğin 32 × 8) azalma, iki boyuttaki yeniden kullanımın ortalamasıyla belirlenir.

### Thread yürütme: warp'lar ve gecikme gizleme {#warp}

Bir blok bir SM'ye atanınca 32 thread'lik warp'lara bölünür; warp, SM'nin zamanlama birimidir. Blokta 256 thread varsa $256/32=8$ warp vardır; SM'de 3 blok varsa 24 warp. Bir SM'de az sayıda skaler işlemci vardır (Tesla'da 8, Fermi'de 32, Kepler'de 192). Neden bu kadar çok warp gerekir? **Gecikme gizleme** için. Bir warp'ın komutu uzun gecikmeli bir işlemin (global bellek okuması) sonucunu bekliyorsa o warp seçilmez, bekleme durumunda olmayan başka bir warp yürütülür. Birden çok warp hazırsa bir öncelik düzeneği birini seçer. Aynı mekanizma boru hatlı kayan nokta aritmetiğinin ve dal komutlarının gecikmesini de gizler. Hazır warp seçimi yürütme zaman çizelgesine boşluk eklemez: **sıfır ek yüklü thread zamanlaması**. CPU'da bir süreç ya da thread değişimi ise yüzlerce çevrim sürer. Uzun gecikmeli işlemleri tolere edebilme yeteneği, GPU'ların önbelleğe ve dal tahminine CPU'lar kadar alan ayırmamasının ana nedenidir.

:::türetim Kaç warp gerekir? (Little yasası)
Bir kuyrukta ortalama bekleyen iş sayısı, varış hızı ile bekleme süresinin çarpımıdır: $L=\lambda W$. Bir SM her çevrimde bir warp komutu verebilsin ve bir bellek okumasının gecikmesi $\ell=400$ çevrim olsun. Her warp okumadan sonra $c$ hesap komutu yürütüp yeniden okuyorsa, bir warp $c$ çevrim çalışır ve sonra $\ell$ çevrim bekler. SM'yi sürekli meşgul tutmak için gereken hazır warp sayısı yaklaşık
$$N_{\text{warp}}\approx\frac{\ell+c}{c}=1+\frac\ell c$$
olur. $c=20$ ise 21 warp gerekir. $c=4$ olan bellek yoğun bir çekirdek 101 warp ister, ki bu SM'nin taşıyabileceğinden (64) fazladır: çekirdek bellek sınırlı kalır. Doluluk (aşağıda) $N_{\text{warp}}$'u sınırlar. Komut düzeyi paralellik (bir thread'in birbirinden bağımsız birden çok okuma yapması) ise daha az warp'la aynı gizlemeyi sağlar.
:::

**Blok boyu alıştırması (GT200).** GT200'de SM başına en çok 1024 thread ve 8 blok vardır, blok başına en çok 512 thread. Matmul için hangi blok boyu seçilmeli?

- **8 × 8 = 64 thread:** SM'yi doldurmak için 16 blok gerekir, ama sınır 8: $8\cdot64=512$ thread, kapasitenin yarısı. Gecikmeyi gizleyecek warp azdır.
- **16 × 16 = 256 thread:** $1024/256=4$ blok, 8 sınırının içinde. Tam kapasite ve en çok warp; iyi bir yapılandırmadır.
- **32 × 32 = 1024 thread:** blok başına 512 sınırını aşar.

Sonuç 16 × 16'dır. Paylaşımlı bellek henüz kullanılmadığı için seçim basitti; genellikle çok daha karmaşıktır.

### Doluluk {#doluluk}

Her CUDA aygıtında yazmaç ve paylaşımlı bellek dahil sınırlı bellek vardır.

- **Yazmaç sınırı**, yüksek başarımla (taşma olmadan) çalışabilecek en çok thread sayısını doğrudan etkiler. G80'de 16 SM'nin her birinde 8192 yazmaç (toplam 131 072) ve SM başına en çok 768 thread vardır. Bu kapasiteyi doldurmak için thread başına yalnız $8192/768\approx10$ yazmaç kalır.
- **Paylaşımlı bellek sınırı**, bir SM'de aynı anda kalabilen blok sayısını etkiler. G80'de SM başına 16 KB vardır. En çok 8 bloğa ulaşmak için blok başına en çok 2 KB kullanılabilir. 16 × 16 karolu matmul, Mds ve Nds için 1'er KB, toplam 2 KB kullanır: 8 blok sığar. Ama asıl sınır 768 thread'tir: $768/256=3$ blok ve paylaşımlı belleğin yalnız 6 KB'ı kullanılır.

**Doluluk**, SM'deki etkin warp sayısının en çok warp sayısına oranıdır. Bir SM'de aynı anda kalabilen blok sayısı dört sınırın en küçüğüdür: thread, blok, yazmaç ve paylaşımlı bellek. Aşağıdaki hesaplayıcı G80'den H100'e bu sınırları uyguluyor. A100'de 256 thread'lik bloklar ve thread başına 64 yazmaçla doluluk %50'dir, sınırlayan yazmaçtır. 48 KB paylaşımlı bellekle %37.5'e düşer. Yüksek doluluk gecikmeyi gizlemeye yardım eder, ama tek başına hedef değildir. Thread başına çok yazmaç kullanıp komut düzeyi paralellikten yararlanan çekirdekler (örneğin yazmaç karolu GEMM) düşük dolulukla en yüksek başarıma ulaşabilir.

::lab l1

### Derleme, uyumluluk ve başlatma {#derleme}

**Çalışma zamanı ve sürücü API'leri.** CUDA C++ iki parçadan oluşur: C++'a küçük bir uzantı kümesi ve bir çalışma zamanı kütüphanesi. Uzantıları içeren her kaynak dosya `nvcc` ile derlenir.

- **Çalışma zamanı API'si** (`cuda` önekli, ör. `cudaMalloc`), ana makinede aygıt belleği ayırma, aktarma ve çok aygıtlı sistemleri yönetme gibi işleri yapar. Örtük başlatma, bağlam ve modül yönetimi sunar; kod sadedir.
- **Sürücü API'si** (`cu` önekli, ör. `cuMemAlloc`), çalışma zamanının altındaki düşük düzeyli C API'sidir. **Bağlamları** (aygıt için süreç karşılığı) ve **modülleri** (dinamik yüklenen kütüphane karşılığı) açıkça gösterir. Çekirdek başlatmak açık fonksiyon çağrıları ister ve daha karmaşıktır. Buna karşılık yalnız gereken modüller yüklü tutulabilir, modüller dinamik olarak yeniden yüklenebilir ve API yalnız cubin nesneleriyle çalıştığı için dilden bağımsızdır.

Çalışma zamanı, bir thread'e bağlam açıkça verilmemişse aygıt başına bir **birincil bağlam** kullanır. Birincil bağlam süreç başına aygıt başına bir tanedir, başvuru sayımlıdır ve sürecin bütün çalışma zamanı kullanıcılarınca paylaşılır. `cudaDeviceSynchronize()` onu eşzamanlar, `cudaDeviceReset()` yok eder. Paylaşımın bir bedeli vardır: aynı süreçteki eklentilerden biri işini bitirip `cudaDeviceReset()` çağırırsa öteki eklentilerin bağlamı habersizce yok olur. Bu yüzden eklentiler sürücü API'siyle kendi bağlamlarını kurabilir; ama bağlamlar aygıt belleği, ek thread'ler ve geçiş maliyeti tüketir. cuBLAS ve cuFFT gibi çalışma zamanı tabanlı kütüphanelerle sürücü API'sini birlikte kullanırken bu paylaşım önemlidir. Uygulamaların çoğu sürücü API'sine gerek duymaz. İki API birlikte çalışabildiği için çalışma zamanını varsayılan kullanıp sürücü API'sini yalnız gereken yerde kullanmak mümkündür.

**nvcc ve PTX.** Çekirdekler, CUDA'nın komut kümesi mimarisi olan **PTX** (Parallel Thread Execution) ile, yani bir tür çevirme diliyle yazılabilir. C++ ile yazmak genellikle daha etkilidir. İki durumda da çekirdekler ikili koda derlenmelidir. `nvcc` bir derleyici sürücüsüdür. **Çevrimdışı derlemede** aygıt kodunu ana makine kodundan ayırır, aygıt kodunu PTX'e ve/veya ikili cubin'e derler. Ana makine kodunda `<<<…>>>` sözdizimini, derlenmiş çekirdekleri yükleyip başlatan çalışma zamanı çağrılarıyla değiştirir ve ana makine derleyicisini çağırır.

**Tam zamanında derleme** (JIT): çalışma sırasında yüklenen PTX kodu aygıt sürücüsünce ikili koda derlenir. Bu, yükleme süresini artırır ama iki yarar sağlar: yeni sürücülerdeki derleyici iyileştirmelerinden yararlanılır, ve uygulama derlendiğinde var olmayan aygıtlarda çalışmanın **tek yolu** budur. Üretilen ikili kod **hesap önbelleğinde** saklanır ve sürücü güncellenince geçersiz olur. NVRTC, CUDA C++'ı çalışma sırasında PTX'e derleyen bir kütüphanedir.

**İkili uyumluluk.** cubin mimariye özgüdür (`-code=sm_80`). Bir alt sürümden sonrakine uyumludur, öncekine ve ana sürümler arasında değildir. Tegra'da ikili uyumluluk yoktur, masaüstü ile Tegra arasında da yoktur.

**PTX uyumluluğu.** Bazı PTX komutları yalnız yüksek hesap yeteneklerinde vardır; `-arch` derleyici seçeneği PTX'e derlerken varsayılan hesap yeteneğini belirler ve en iyi başarım için her zaman verilmelidir. Belirli bir hesap yeteneği için üretilmiş PTX, eşit ya da daha yüksek her hesap yeteneğinin ikilisine derlenebilir. Ama eski PTX'ten üretilen ikili, yeni donanım özelliklerini kullanamaz. Örneğin Pascal (6.0) için üretilmiş PTX'ten Volta'ya (7.0) derlenen ikili Tensor Core komutlarını kullanmaz.

**Uygulama uyumluluğu.** Uygulamaya hangi PTX ve ikililerin gömüleceği `-gencode` ile denetlenir:

```text
nvcc x.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=\"compute_70,sm_70\"
```

Bu örnek 5.0 ve 6.0 ikililerini, 7.0 için hem ikiliyi hem PTX'i gömer. Çalışma sırasında en uygun kod seçilir:

- 5.0 ve 5.2 aygıtlarında 5.0 ikilisi;
- 6.0 ve 6.1'de 6.0 ikilisi;
- 7.0 ve 7.5'te 7.0 ikilisi;
- 8.0 ve 8.6'da PTX'ten JIT.

Aygıt kodunda `__CUDA_ARCH__` makrosu (ör. `-arch=compute_80` ile 800) mimariye özgü kod yollarını ayırır. Örneğin warp indirgeme komutları (`__reduce_add_sync`) yalnız 8.0 ve sonrasında vardır.

:::uyarı Kaynak düzeltmeleri (lect_06 slayt 14 ve 16)
**Slayt 14.** "X.y için üretilmiş cubin yalnız z > y olan X.z aygıtlarında çalışır" diyor. Doğrusu **z ≥ y**'dir: cubin kendi hesap yeteneğinde elbette çalışır. Örneğin `sm_80` ikilisi 8.0 ve 8.6'da çalışır.

**Slayt 16.** Warp shuffle fonksiyonlarını "5.0 ve sonrası" için örnek veriyor. Bu ifade NVIDIA'nın programlama kılavuzundaki bir örnekten gelir, ama shuffle komutları ilk kez **hesap yeteneği 3.0** (Kepler) ile geldi. Kavram olarak doğru olan şudur: bir özelliği kullanan kod en az o özelliğin geldiği `-arch` ile derlenmelidir.
:::

**Başlatma ve zaman ölçümü.** CUDA 12.0'dan beri `cudaInitDevice()` ve `cudaSetDevice()` çalışma zamanını ve aygıtın birincil bağlamını başlatır. Bu çağrılar olmadan çalışma zamanı örtük olarak aygıt 0'ı kullanır ve ilk API çağrısında kendini başlatır. Daha önce bunu yalıtmak için etkisiz `cudaFree(0)` çağrısı kullanılırdı. Bağlam yaratılırken gerekirse aygıt kodu JIT ile derlenip aygıt belleğine yüklenir. GPU kodunu ölçerken iki şey sağlanmalıdır:

1. GPU başlatılmış olmalıdır, yani soğuk çalıştırma ölçülmemelidir.
2. Ölçülen iş gerçekten bitmiş olmalıdır. Çekirdek başlatmaları **eşzamansızdır**: başlatma hemen döner. Saati okumadan önce `cudaDeviceSynchronize()` çağrılmazsa "mucizevi derecede kısa" süreler ölçülür. `cudaEvent` çiftleri GPU tarafında ölçüm yapar.

### Bellek ayırma, aktarma ve hizalama {#aktarma}

**Ayırma ve aktarma.** `cudaMalloc((void**)&d, bytes)` aygıt belleği ayırır. İlk parametre, ayrılan adresin yazılacağı işaretçi değişkeninin **adresidir**; ikinci parametre bayt cinsinden boyuttur. `cudaFree(d)` belleği bırakır. `cudaMemcpy(hedef, kaynak, bayt, tür)` belleği taşır; tür `cudaMemcpyHostToDevice`, `DeviceToHost`, `DeviceToDevice` ya da `HostToHost`'tur. `cudaMemcpy` eşzamanlıdır, yani işlem bitene kadar bloklar. `cudaMemcpyAsync` ise akışlarla (streams) eşzamansız çalışır.

:::uyarı Kaynak düzeltmesi (lect_05 slayt 69): ana makine yerel belleğe erişemez
Slayt "global, yerel ve sabit belleğin ana makinece API çağrılarıyla okunup yazılabildiğini" söylüyor. Ana makinenin API'yle erişebildiği **global** ve **sabit** bellektir (`cudaMemcpy`, `cudaMemcpyToSymbol`). Yerel bellek fiziksel olarak aygıt belleğinde durur, ama her thread'e özeldir. Adresi ana makineye görünmez ve çekirdek bitince yok olur.
:::

**Hizalama ve pitch.** Global bellekteki bir değişkenin ya da ayırma rutinlerinin döndürdüğü her adres en az **256 bayta hizalıdır**. Bu GPU'ya özgüdür; FPGA ve CPU'ların gereksinimleri farklıdır. 2-B bir dizide başlangıç adresi hizalıdır, ama sonraki satırların başı satır genişliğine bağlıdır. Bu yüzden `cudaMallocPitch` (2-B) ve `cudaMalloc3D` (3-B) satırları dolgulayıp bir **pitch** (adım) döndürür:

| genişlik | pitch (256 bayt hizalama) |
|---|---|
| 5, 50, 250 bayt | 256 |
| 500 bayt | 512 |
| 700 bayt | 768 |

Erişim `(float*)((char*)ptr + r * pitch)` ile yapılır. Kopyalama `cudaMemcpy2D`/`cudaMemcpy3D` ile yapılır; ana makine tarafının pitch'i genellikle `Ncols * sizeof(tip)`'tir.

**Doğal hizalama.** Global bellek komutları thread başına 1, 2, 4, 8 ya da 16 baytlık sözcükleri destekler. Bir erişim, ancak tür bu boylardan biriyse ve adres boyun katıysa (doğal hizalı) **tek komuta** derlenir. Aksi hâlde tam birleştirilemeyen çok sayıda komuta bölünür. Yerleşik türler (`float2`, `float4` dahil) bunu kendiliğinden sağlar. Yapılarda `__align__(8)` ya da `__align__(16)` kullanılır. Bir tuzak, birden çok `cudaMalloc` yerine tek büyük bir bloğun dizilere bölünmesidir. Bir float dizisinin ardından gelen float2 dizisi 8 bayta hizalı olmayabilir; doğal hizalı olmayan 8 ya da 16 baytlık okumalar **yanlış sonuç** (birkaç sözcük kaymış veri) verir.

### Global bellekte birleştirme {#birlestirme}

Global belleğe 32, 64 ya da 128 baytlık hizalı işlemlerle (transaction) erişilir; çağdaş GPU'larda temel birim 32 baytlık **sektördür**. "11 bayt okumak" diye bir şey yoktur, 32 bayt okunur; 33 bayt iki sektör eder. Bir warp global belleğe erişen bir komut yürütünce, thread'lerin erişimleri **birleştirilir** (coalescing): warp'ın dokunduğu farklı sektörler kadar işlem yapılır. Ne kadar çok işlem gerekirse o kadar çok kullanılmayan sözcük taşınır ve verim düşer.

Ders bunu iki uçla gösteriyor. 32 thread ardışık ve 128 bayta hizalı adreslerden 4'er bayt okursa tek bir 128 baytlık işlem yeter. Her thread'in 4 baytlık erişimi için ayrı bir 32 baytlık işlem gerekirse $32\cdot32=1024$ bayt taşınır: verim **8'de 1**. 2-B erişimde `(tx, ty)` thread'i `BaseAddress + width * ty + tx` adresine eriştiğinde tam birleştirme için iki koşul gerekir: bloğun genişliği ve dizinin genişliği warp boyunun katı olmalıdır. Genişliği bunun katı olmayan diziler, satırları dolgulanırsa çok daha verimli okunur; `cudaMallocPitch` bunu yapar. Birleştirmenin kalıbı M7.1'deki deneyde ayrıntılı ölçülüyor.

<figure class="fig wide"><svg viewBox="0 0 640 222" role="img" aria-label="Bir warp'ın farklı erişim desenlerinde dokunduğu 32 baytlık bellek sektörü sayısı ve bant genişliği verimi"><text x="220" y="18" class="sm">sektör (32 B)</text><text x="520" y="18" class="sm">verim</text><text x="210" y="48" text-anchor="end" class="sm">ardışık, hizalı: A[tid]</text><rect x="220" y="34" width="32" height="18" class="cp-f"/><text x="258" y="48" class="sm">4</text><text x="520" y="48" class="sm">100%</text><text x="210" y="78" text-anchor="end" class="sm">1 öğe kaymış: A[tid + 1]</text><rect x="220" y="64" width="40" height="18" class="ca-f"/><text x="266" y="78" class="sm">5</text><text x="520" y="78" class="sm">80%</text><text x="210" y="108" text-anchor="end" class="sm">adım 2: A[2·tid]</text><rect x="220" y="94" width="64" height="18" class="ca-f"/><text x="290" y="108" class="sm">8</text><text x="520" y="108" class="sm">50%</text><text x="210" y="138" text-anchor="end" class="sm">adım 32: A[32·tid] (sütun)</text><rect x="220" y="124" width="256" height="18" class="cb-f"/><text x="482" y="138" class="sm">32</text><text x="520" y="138" class="sm">12%</text><text x="210" y="168" text-anchor="end" class="sm">rastgele: A[perm[tid]]</text><rect x="220" y="154" width="248" height="18" class="cb-f"/><text x="474" y="168" class="sm">31</text><text x="520" y="168" class="sm">13%</text><text x="20" y="196" class="sm">32 thread × 4 bayt = 128 yararlı bayt. Verim = 128 / (32 × sektör sayısı).</text><text x="20" y="212" class="sm">Bellek her zaman hizalı 32 baytlık sektörler hâlinde okunur.</text></svg><figcaption>Şekil 1 — Birleştirme: ardışık hizalı erişim 4 sektör (%100), bir öğe kayma 5 sektör (%80), adım 2 8 sektör (%50), sütun erişimi 32 sektör (%12.5), rastgele erişim bu örnekte 31 sektör: bant genişliğinin yaklaşık sekizde biri (hesaplandı).</figcaption></figure>

### Paylaşımlı bellek ve bank çakışmaları {#bank}

Paylaşımlı bellek yonga üstündedir. Bant genişliği global ve yerel bellekten çok daha yüksek, gecikmesi çok daha düşüktür. Yüksek bant genişliği için eşit boyutlu **bank**'lara bölünmüştür: ardışık 4 baytlık sözcükler ardışık bank'lara düşer ve 32 bank vardır. $n$ farklı bankta $n$ adrese yapılan bir istek aynı anda hizmet alır; bant genişliği tek bankınkinin $n$ katıdır.

İki adres **aynı bankta farklı sözcüklere** düşerse çakışma vardır. Donanım isteği çakışmasız alt isteklere böler ve bant genişliği bu sayı kadar azalır; buna **$n$ yollu bank çakışması** denir. Aynı sözcüğe erişen thread'ler çakışmaz: o sözcük **yayınla** dağıtılır, yazmada thread'lerden biri kazanır. Global bellekten farklı olarak burada önemli olan adreslerin sektörlere değil bank'lara dağılımıdır.

:::türetim Adımlı erişim ve dolgu
Thread $t$, $A[s\cdot t]$'ye (4 baytlık öğeler) erişsin. Bankı $(s\,t)\bmod32$'dir. Bir bankta kaç thread toplandığı $g=\gcd(s,32)$ ile belirlenir: $s\,t\equiv s\,t'\pmod{32}$ ancak $t\equiv t'\pmod{32/g}$ ise olur. 32 thread $32/g$ farklı banka düşer, her bankta $g$ farklı sözcük vardır: **$g$ yollu çakışma**. Tek adımlarda $g=1$, çakışma yoktur. $s=2$'de 2 yollu, $s=32$'de 32 yollu çakışma oluşur.

$32\times32$'lik bir karo `float tile[32][32]`, **sütun** boyunca okunursa (`tile[tid][c]`) adım 32'dir: bütün thread'ler aynı bankta, 32 yollu çakışma. Bu, matris devriği gibi satırı yazıp sütunu okuyan çekirdeklerde olur. Çözüm bir sütunluk dolgudur: `float tile[32][33]`. Adım 33 olur, $\gcd(33,32)=1$ ve çakışma yok. Bedel, karo başına 32 float (128 bayt) boş bellektir.
:::

:::uyarı Kaynak düzeltmesi (lect_06): 8 baytlık bank'lar "yeni mimarilerin" değil, Kepler'indir
Slayt "yeni mimarilerde bank genişliği 8 bayttır" diyor. Ayarlanabilir 8 baytlık bank kipi yalnız **Kepler**'de (hesap yeteneği 3.x) vardı. Maxwell'den bu yana bank'lar 4 bayt genişliğindedir. 64 ve 128 bitlik erişimlerde istek donanımca yarım ya da çeyrek warp'lara bölünür. Ardışık `double` ya da `float4` okumaları yine çakışmasızdır.
:::

Aşağıdaki çözümleyici bir warp'ın adreslerini bank'lara eşleyip çakışma derecesini sayıyor. `[32][32]` karoda sütun okuma 32 geçiş, `[32][33]` ile 1 geçiş. Adım 2 ile 2 yollu çakışma, yayında tek geçiş, ardışık `double`'da 2 (ideal) geçiş görülüyor.

::lab l2

### Hata denetimi {#hata}

Bütün çalışma zamanı fonksiyonları bir hata kodu döndürür. Ama eşzamansız bir fonksiyon, aygıt işi bitirmeden döndüğü için aygıtta oluşabilecek eşzamansız hataları bildiremez; yalnız işi başlatmadan önce ana makinede oluşan hataları (genellikle parametre doğrulama) bildirir. Eşzamansız bir hata, sonraki ilgisiz bir çalışma zamanı çağrısınca raporlanır. Eşzamansız bir çağrıdan hemen sonra hatayı görmenin tek yolu `cudaDeviceSynchronize()` çağırıp onun hata kodunu denetlemektir.

Çalışma zamanı, ana makinenin her thread'i için `cudaSuccess` ile başlayan bir hata değişkeni tutar. Bir hata olduğunda (parametre ya da eşzamansız) değişkenin üzerine yazılır:

- `cudaPeekAtLastError()` bu değişkeni döndürür.
- `cudaGetLastError()` değişkeni döndürür ve `cudaSuccess`'e sıfırlar.

Çekirdek başlatmaları hata kodu döndürmez. Başlatma öncesi hataları almak için başlatmadan hemen sonra `cudaPeekAtLastError()` ya da `cudaGetLastError()` çağrılmalıdır. Hatanın önceki çağrılardan gelmediğinden emin olmak için de başlatmadan hemen önce `cudaGetLastError()` ile değişken sıfırlanmalıdır. Çekirdeğin çalışırken oluşan hatalarını görmek için arada eşzamanlama gerekir. Yaygın kalıp, her çağrıyı saran bir `CUDA_CHECK(x)` makrosu ve başlatmadan sonra `CUDA_CHECK(cudaGetLastError())`'dır.

### Dal ayrışması ve tahminleme {#ayrisma}

Herhangi bir akış denetimi komutu (`if`, `switch`, `do`, `for`, `while`) aynı warp'ın thread'lerini farklı yürütme yollarına saptırabilir; buna **ayrışma** denir. Bu durumda yollar **sırayla** yürütülür, yolda olmayan thread'ler maskelenir. Warp'ın yürüttüğü toplam komut sayısı artar, bütün yollar bitince thread'ler yeniden aynı yolda birleşir.

Denetim akışı thread kimliğine bağlıysa koşul, ayrışan warp sayısını en aza indirecek biçimde yazılmalıdır. Bu mümkündür, çünkü warp'ların blok içindeki dağılımı deterministiktir. En basit örnekte koşul yalnız `threadIdx / 32`'ye bağlıdır: koşul warp'lara tam hizalı olduğu için hiçbir warp ayrışmaz. Aşağıdaki deneyde `if (tid % 2 == 0)` her warp'ı ayrıştırıp SIMT verimini %50'ye düşürüyor; `(tid / 32) % 2` %100 veriyor. `if (tid < k)` yalnız k'yi içeren warp'ı ayrıştırıyor. Değişken sayılı döngüde warp en uzun thread'i bekliyor.

::lab l3

**Dal tahminlemesi (predication) ≠ dal tahmini (prediction).**

- **CPU'da dal tahmini:** koşullu bir dalın alınıp alınmayacağı tahmin edilir ve tahmine göre spekülatif yürütmeye geçilir. Koşul hesaplanınca tahmin doğruysa devam edilir, yanlışsa spekülatif iş atılıp doğru daldan başlanır.
- **GPU'da tahminleme:** koşula bağlı komutların hiçbiri atlanmaz. Her komut, koşula göre doğru ya da yanlış olan thread başına bir **koşul koduyla** (yüklemle, predicate) ilişkilendirilir. Hepsi zamanlanır, ama yalnız yüklemi doğru olanlar gerçekten yürür. Yanlış yüklemli komutlar sonuç yazmaz, adres hesaplamaz ve işlenen okumaz.

Derleyici kısa `if` gövdelerini dallanma yerine tahminlemeyle derler: ayrışma ve dal maliyeti olmaz, iki yolun komutları da (maskeli) yürür. Dal eniyilemesi, **komut boru hatlarının** dallanmayla kesintiye uğramasını önlemek için önemlidir.

### Döngü açma ve sayaçlar {#acma}

Derleyici, tur sayısı bilinen küçük döngüleri kendiliğinden açabilir. Programcı bunu `#pragma unroll` ile denetler; yönerge döngüden hemen önce yazılır ve yalnız o döngüye uygulanır:

- `#pragma unroll 5` döngüyü 5 kat açar; derleyici doğruluğu güvenceye alan kodu ekler (örneğin $n<5$ ise yalnız $n$ tur).
- `#pragma unroll 1` açmayı tamamen önler.
- Sayısız `#pragma unroll`, tur sayısı sabitse döngüyü tamamen açar, değilse hiç açmaz.

Açma, döngü sayacı ve dal komutlarını azaltır, bağımsız komutları yan yana getirerek komut düzeyi paralelliği artırır. Gövde ne kadar küçükse kazanç o kadar büyüktür. En iyi açma derecesini bulmak programcıya kalır. Döngü sayaçları her zaman pozitif olsa bile **işaretli** (`int`) tanımlanmalıdır. İşaretli taşma tanımsız olduğu için derleyici, `out[i] = in[offset + stride*i]` gibi adres hesaplarını daha saldırgan biçimde eniyileyebilir. İşaretsiz sayaçta sarmalanma tanımlı olduğu için bu dönüşümler yapılamaz.

### Özet {#ozet}

- GPU hesaba, CPU kontrol ve önbelleğe silisyum ayırır. SIMT (warp = 32 thread) programlama modelidir. CUDA yaklaşımı yedi adımdır: çözümle → veri paralel kısmı bul → kopyala → hesapla → geri kopyala → doğrula (kayan nokta birleşmeli değildir) → yinele.
- Hiyerarşi grid, blok ve thread'dir (`threadIdx`, `blockIdx`, `blockDim`). Bloklar bağımsızdır ve ölçeklenebilirliği sağlar; blok içi işbirliği `__shared__` ve `__syncthreads()` ile yapılır. Kümeler Hopper ile gelmiştir. Volta'dan beri thread'ler bağımsız zamanlanır.
- Bellek türleri yazmaç, yerel (taşma), paylaşımlı, global, sabit (yayın) ve doku (2-B önbellek, süzme, adresleme kipleri) belleğidir. CGMA 1 G80'de tepenin %5.9'udur; $T\times T$ karo trafiği $T$ kat azaltır.
- Warp'lar gecikmeyi sıfır ek yüklü zamanlamayla gizler (Little yasası: $1+\ell/c$ warp). Doluluk thread, blok, yazmaç ve paylaşımlı bellek sınırlarının en darıdır.
- Derleme ve uyumluluk: çalışma zamanı ve sürücü API'si, birincil bağlam, PTX ve cubin, JIT, `-gencode`, `__CUDA_ARCH__`. Çekirdekler eşzamansızdır; ölçmeden önce eşzamanla.
- Global bellekte 256 bayt hizalama, pitch ve doğal hizalama kuralları geçerlidir. Birleştirmede 32 baytlık sektörler sayılır; dağınık erişim 8 kat kaybettirir.
- Paylaşımlı bellekte 32 bank vardır; adım $s$ ile çakışma derecesi $\gcd(s,32)$'dir, `[32][33]` dolgusu sütun okumayı çakışmasız yapar.
- Hata denetimi: Peek/GetLastError ve eşzamanlama. Ayrışmayı azaltmak için koşul warp'a hizalanır; kısa dallar tahminlemeyle derlenir. `#pragma unroll` ve işaretli sayaçlar eniyilemeye yardım eder.
