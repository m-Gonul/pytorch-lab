### Neden paralellik? {#neden}

**Moore yasası.** Gordon Moore'un 1965'teki gözlemine göre bir tümleşik devredeki transistör sayısı düzenli olarak ikiye katlanıyordu. 1975'te bu süreyi "yaklaşık iki yıl" olarak düzeltti ve yasa bu biçimiyle bilinir. Gözlem transistör **sayısı** hakkındadır, saat **frekansı** hakkında değil. Saat frekansı, bir işlemcinin saniyede kaç işlem yapabildiğiyle yakından ilgili bir özelliktir. Ama her işlem aynı sayıda saat çevrimi sürmez; yalnız frekansa bakarak yongaları karşılaştırmak yanıltıcıdır.

**Frekans duvarı.** Uzun süre transistör sayısındaki artışa frekans artışı da eşlik etti. Ancak yüksek frekans daha çok ısı demekti. Tasarımcılar çalışma gerilimini düşürerek tepki verdiler, ama bu sorunu tam çözmeye yetmedi. Saat frekansları 2004–2005 civarında 2–4 GHz aralığında doydu.

:::türetim Frekans yerine çekirdek: güç hesabı
CMOS devrelerin dinamik gücü yaklaşık $P\approx C\,V^2f$'dir: $C$ anahtarlanan kapasite, $V$ gerilim, $f$ frekans. Bir transistörün daha hızlı anahtarlanması daha yüksek gerilim ister ($V\propto f$ kabaca); bu yüzden $P\propto f^3$. Frekansı %20 artırmak gücü $1.2^3=1.73$ katına çıkarır, başarımı ise en iyi ihtimalle 1.2 katına.

Tersini düşünelim. Aynı alana iki çekirdek koyup ikisini de %20 düşük frekansta ve gerilimde çalıştıralım. Güç $2\cdot0.8^3=1.02$ kat olur, yani neredeyse aynı. Toplam iş kapasitesi ise $2\cdot0.8=1.6$ kattır. Koşul, işin iki çekirdeğe bölünebilmesidir.

Dennard ölçeklemesi (transistör küçüldükçe güç yoğunluğunun sabit kalması) 2000'lerin ortasında sızıntı akımları yüzünden sona erdi. Artan transistörlerin gidebileceği tek yer daha çok çekirdek ve daha geniş vektör birimleri oldu. Bunun bedelini yazılım öder: başarım artık kendiliğinden gelmez, paralel kod yazılmalıdır.
:::

**Çok çekirdekli işlemciler ve GPGPU.** Daha fazla hesap gücü talebini karşılamanın tek yolu bir yongaya daha çok hesap mantığı ve çekirdek sığdırmaktı. IBM'in 2001'deki Power4'ü tek kalıpta (die) ilk çift çekirdekli işlemciydi. O zamandan beri çok çeşitli çok çekirdekli yongalar çıktı. Bunlar 1990'ların ortası ve sonundaki çok soketli platformların doğal evrimiydi: her biri ayrı yongada tek çekirdekli birkaç CPU barındıran makineler.

Beklenmedik olan, **GPGPU**'nun yükselişi oldu: grafik işlem birimiyle (GPU) genel amaçlı hesap. Bir GPU çekirdeği çağdaş bir CPU çekirdeğinin yanında zayıftır. Ama GPU'lar yüksek bant genişlikli, hızlı belleğe bağlı yüzlerce ya da binlerce çekirdekten oluşan devasa paralel mimarilerdir. Sonuç mertebeler hızlı hesaptır. GPU'nun bir avantajı daha vardır: **watt başına işlem** (GFLOP/W) üstündür, yani harcanan enerji başına daha çok hesap yapar. Bu özellikle sunucu ve bulut altyapısında kritiktir: bir CPU'nun ömrü boyunca tükettiği enerjinin maliyeti fiyatını aşabilir.

Bu başarım bedava değildir:

- Çok çekirdekli işlemciler çok iş parçacıklı (multi-threaded), paralel kod ister.
- Geleneksel olarak deterministik bir adım dizisi izleyen algoritmaların açıkça yeniden tasarlanması gerekir.
- Yeni API'ler gerekir: CUDA, OpenCL.

### Paralel makinelerin sınıflandırması {#flynn}

Birden çok kaynakla başarım arayışı 1960'lara uzanır, bu yüzden paralel makinelerin mimarisini betimlemenin bir yolu gerekti. Michael Flynn 1966'da makineleri iki ölçüte göre sınıfladı: aynı anda kaç veri öğesi işleyebildikleri ve aynı anda kaç farklı komut yürütebildikleri.

| | tek veri | çok veri |
|---|---|---|
| **tek komut** | SISD: sıralı makine | SIMD: her komut bir veri kümesine uygulanır (vektör işlemciler) |
| **çok komut** | MISD: olağan koşullarda anlamsız; hata toleransında aynı veriyi birkaç makine işler ve çoğunluk kararı alınır (askerî, havacılık) | MIMD: en esnek sınıf; çok çekirdekli makineler |

Sınıflandırma yıllar içinde, özellikle MIMD altında, alt kategorilerle zenginleşti. MIMD iki geniş gruba ayrılır:

- **Paylaşımlı bellekli MIMD.** Bütün işlemcilerin erişebildiği tek bir bellek uzayı vardır. Paylaşımlı bellek CPU'lar arasındaki etkileşimi en az ek yükle sağlar, ama aynı zamanda sistemin ölçeklenmesini sınırlayan bir darboğazdır. Bir çözüm belleği işlemciler arasında bölmektir. Her CPU belleğin bir kısmına "sahip" olur: yerel belleğine hızlı, diğer CPU'ların belleğine daha yavaş erişir, ama adresleme şeması evrenseldir. Bu tasarım **NUMA**'dır (Non-Uniform Memory Access) ve paylaşımlı bellekli makinelerin birkaç on CPU'ya ölçeklenmesini sağlar. Paylaşımlı bellekli makineler ikiye ayrılır:
    - **Simetrik çok işlemcili** (SMP) platformlarda bütün CPU'lar eşdeğerdir ve sistem yazılımı dahil her programı çalıştırabilir.
    - **Efendi–işçi** düzeninde bazı işlemciler özel yazılıma ayrılır, yani yardımcı işlemcidir. GPU'lu sistemler bu kategoriye benzer. Ama yüksek başarımlı GPU platformlarının çoğunda CPU ve GPU'nun bellek uzayları ayrıdır. Bu yüzden veri taşımanın karmaşıklığını gizleyen sürücü yeniliklerine rağmen bunlar paylaşımlı bellekli platform sayılmaz.
- **Dağıtık bellekli** (hiçbir şey paylaşmayan) **MIMD.** İşlemciler mesaj alışverişiyle haberleşir. İletişim pahalıdır, ama paylaşılan tek bir ortam olmadığı için bu makineler yer ve enerji dışında pratik bir sınır olmadan ölçeklenebilir.

| özellik | paylaşımlı bellekli MIMD | dağıtık bellekli MIMD |
|---|---|---|
| bellek modeli | küresel ortak adres uzayı | her işlemcinin özel yerel belleği |
| iletişim | paylaşılan değişkenler | mesaj geçirme (MPI vb.) |
| örnekler | çok çekirdekli CPU'lar, SMP sunucular | HPC kümeleri, bulut kümeleri |
| ölçeklenme | sınırlı (onlarca–yüzlerce çekirdek) | çok yüksek (binlerce–milyonlarca çekirdek) |
| programlama | daha kolay | daha zor (veri dağıtımı açıkça yapılır) |
| darboğaz | bellek bant genişliği, önbellek tutarlılığı | ağ gecikmesi, iletişim maliyeti |

Paylaşımlı bellekli MIMD tek sunucular ve sıkı bağlı makineler için, dağıtık bellekli MIMD ise süperbilgisayarlar, kümeler ve veri merkezleri boyunca büyük ölçekli yapay zekâ eğitimi (M7.4) için uygundur.

**Modern işlemciler** Flynn'in kategorileri arasındaki sınırları bulanıklaştırır: hangi düzeyde bakıldığına göre hem MIMD hem SIMD'dirler. Bir GPU, akış çok işlemcileri (SM) düzeyinde MIMD'dir (her SM kendi programını çalıştırabilir), bir SM'nin içinde ise SIMD'dir (NVIDIA'nın deyimiyle SIMT: bir warp'ın 32 thread'i aynı komutu yürütür; M7.1). Küçülen transistörlerin getirdiği bütçeyi kullanmanın iki eğilimi vardır:

1. **Çekirdek sayısını artırmak,** genişletilmiş SIMD komut kümeleri (SSE ve ardılları, AVX, AES-NI) ve daha büyük önbelleklerle birlikte. Örnekleri Intel x86 ve Xeon Phi'dir.
2. **Farklı türden çekirdekleri aynı pakette birleştirmek:** genellikle her biri farklı bir iş türüne eniyilenmiş CPU ve GPU çekirdekleri. Örnekleri AMD'nin APU'ları ve Intel'in tümleşik grafikli CPU'larıdır.

### GPU ve CPU {#gpu-cpu}

GPU'lar (grafik hızlandırıcı kartlar), büyük miktarda grafik verisini ekran tamponuna yazılmadan önce çok hızlı işlemek için geliştirildi. Bu amaç, geleneksel CPU'lardan farklı bir yerleşim dayattı.

| CPU | GPU |
|---|---|
| büyük (bazen çok düzeyli) yonga üstü önbellekler | küçük önbellekler |
| az sayıda karmaşık (boru hatlı) aritmetik-mantık birimi | çok sayıda basit, paralel çalışan aritmetik birim (grafikte veri yeniden kullanımı az, programlar görece basit) |
| ana bellekten veri beklerken durmamak için karmaşık komut çözme ve dal tahmini donanımı | çok sayıda çekirdeği beslemek için çok geniş, hızlı bellek yolları |

CPU kalıbında önbellek alanı baskındır, GPU'da hesap mantığı. Bu farkın arkasında iki farklı amaç vardır. **Gecikmeye eniyilenmiş** işlemci (CPU) tek bir işi olabildiğince çabuk bitirir. **Verime eniyilenmiş** işlemci (GPU) birim zamanda olabildiğince çok iş bitirir: tek bir iş daha geç bitebilir, ama aynı anda binlercesi yürür. GPU bellek gecikmesini önbellekle değil, bekleyen thread'lerin yerine hazır olanları çalıştırarak gizler (M10.4).

**PCIe eşiği.** Ayrık bir GPU ana CPU ile PCIe gibi görece yavaş bir yolla haberleşir. Veri işlenmeden önce ana bellekten GPU belleğine, sonuçlar da geri taşınmalıdır. Bu bir **veri boyutu eşiği** yaratır ve bu eşiğin altında GPU etkili değildir. Örneğin PCIe 4.0 x16 ile yaklaşık 25 GB/s'lik bir yolda 100 MB'lık bir vektörü taşımak 4 ms sürer. Bu vektör üzerinde tek bir toplama yapmak ise GPU belleğinin 1–2 TB/s'lik bant genişliğiyle 0.1 ms'nin altındadır. Taşıma, ancak veri GPU'da çok sayıda işlemden geçecekse karşılığını verir. Etkili bir sistem, işletim sistemini ve sıralı, gecikmeye duyarlı işleri çalıştıran CPU'yu, yüksek verimli işi yapan bir yardımcı işlemci olarak GPU ile birleştirir. CPU ve GPU çekirdeklerinin aynı bellek uzayını paylaşmasının neden önemli olduğu buradan anlaşılır.

#### Etkili işlemciler {#islemciler}

Ders, paralel mimarinin farklı yollarını gösteren beş işlemciyi inceliyor.

- **Cell BE** (Sony, Toshiba, IBM). PlayStation 3'ü güçlendiren işlemcidir: tek yonga üzerinde efendi–işçi, heterojen bir MIMD makine.
    - Efendi, iki thread çalıştırabilen 64 bitlik bir PowerPC çekirdeğidir (PPE). İşletim sistemini çalıştırır ve işçileri yönetir.
    - İşçiler 128 bitlik vektör işlemcileridir (SPE). Her SPE, veri ve kodu için önbellek olmayan 256 KB'lık bir **yerel depoya** sahiptir.
    - SPE'ler PPE ile ikili uyumlu değildir; SIMD için kendi komut kümeleri vardır. Birbirleriyle yüksek hızlı bir halka (EIB) üzerinden haberleşir. Ana belleğe doğrudan erişemezler, DMA aktarımıyla veri çekerler.
    - Donanım en yüksek hesap verimi için, programlama kolaylığı pahasına tasarlandı ve programlanması en zor platformlardan biri olarak ün yaptı.
    - PS3 kümeleri astrofizik, uydu görüntüleme ve biyomedikal uygulamalar için "ucuz süperbilgisayar" oldu. 2008–2009'da dünyanın en hızlısı olan IBM Roadrunner 12 240 PowerXCell 8i ve 6 562 AMD Opteron içeriyordu. Cell'in geliştirilmesi durdu; büyük olasılıkla programlama zorluğunun ve GPU hesabındaki ilerlemelerin kurbanı oldu.
- **NVIDIA Kepler** (2012), NVIDIA'nın özellikle hesap için tasarladığı üçüncü GPU mimarisidir.
    - Çekirdekler akış çok işlemcilerinde (SMX) gruplanır. Her SMX'te SIMD biçiminde yürüyen 192 tek duyarlıklı çekirdek ve 64 çift duyarlıklı birim vardır; her SMX kendi programını çalıştırabilir. SMX sayısı aynı ailedeki yongaları ayıran ana etkendir.
    - GTX Titan'da 15 SMX'ten biri üretim verimi için kapatılmıştır: $14\cdot192=2688$ çekirdek, tek duyarlıkta 4.5 TFLOPS. Çift duyarlıkta bunun yaklaşık üçte biri elde edilir, çünkü DP birimleri SP'nin üçte biridir. İki GPU'lu Titan Z'de bütün SMX'ler açıktır: $2\cdot15\cdot192=5760$ çekirdek.
    - Bir L2 önbelleği gecikmeyi azaltır. Birleşik L1/paylaşımlı bellek, CPU'ların çekirdek başına MB düzeyindeki önbelleklerinin yanında küçüktür. Paylaşımlı bellek önbellek gibi uygulamaya saydam değildir: adreslenebilir, kullanıcının yönettiği bir önbellektir. Önceki GPU'larda önbellek yoktu, paylaşımlı bellek SM başına 16 KB'tı.
    - **Dinamik paralellik:** bir çekirdeğin (kernel) ek iş üretebilmesi. Kabaca özyineleme gibi düşünülebilir, ama GPU'da çalışabilecek kodun karmaşıklığını çok daha fazla artırır.
    - Sistem en yüksek saat hızında üretilen ısıyı kaldırabiliyor ve iş tamamen paralelse GPU'lar watt başına çok iyi işlem verir.
- **AMD APU'lar** (PlayStation 4 de bunlardan birini kullanır) CPU ve GPU çekirdeklerini aynı kalıpta birleştirir. Asıl önemli nokta bellek uzaylarının birleşmesidir. GPU'ya iş atamanın iletişim ek yükü ve sonuçları almanın gecikmesi yoktur; GPU programlamanın en büyük zahmetlerinden biri olan açık ya da örtük veri aktarımları ortadan kalkar. APU'lar, HSA Vakfı'nın (AMD, ARM, Imagination, MediaTek, TI, Samsung, Qualcomm) geliştirdiği Heterojen Sistem Mimarisini (HSA) uygular. HSA iki çekirdek türü tanımlar:
    - **LCU** (gecikme hesap birimi): CPU'nun genelleştirilmesidir; hem kendi komut kümesini hem HSA ara dilini (HSAIL) destekler.
    - **TCU** (verim hesap birimi): GPU'nun genelleştirilmesidir; yalnız HSAIL'i destekler.
- **Tilera TILE-Gx8072.** GPU'lar yaklaşık on yıldır yüzlerce basit çekirdek barındırıyordu, ama yalnız grafik işlerine özgü koşullarda etkiliydiler. Aynı başarıyı işletim sistemi ve uygulama yazılımı çalıştırabilen genel amaçlı bir CPU için göstermek başka bir iştir. Bunu başaran tasarımlar, geçmişte paralel makine kurmakta kullanılan ağ yapılarını silikona taşıdı; bir minyatürleştirme başarısıdır.
    - Çok çekirdekli paradigmanın ilk örneği 2007'de 2-B ızgarada 64 çekirdekli TILE64 oldu; sonra 9–72 çekirdekli sürümler geldi.
    - Gx8072'de çekirdekler, toplam bant genişliği 110 Tbps'yi aşan beş bağımsız ağdan oluşan 2-B bir ızgarayla (iMesh) bağlıdır; atlama başına bir saat çevrimidir.
    - Her çekirdeğin 32 KB veri ve 32 KB komut L1'i ve 256 KB L2'si vardır; 18 MB tutarlı L3 paylaşılır.
    - Ağ işlemeye (32 × 1 Gbps ve 8 × 10 Gbps bağlantı noktası, sıkıştırma ve şifreleme motorları), çoklu ortama ve buluta yönelir. Hem yardımcı işlemci olarak hem de Linux çalıştıran bağımsız bir platform olarak kullanılabilir. Geliştirme ortamı GNU araçları, Eclipse, Boost ve TBB gibi açık kaynak araçlar üzerine kuruludur.
    - Yeni bir işlemciyle karşılaşınca sorulacak en önemli sorular şunlardır: Bunu nasıl programlarım? Hangi derleyici? Hangi çerçeveler?
- **Intel Xeon Phi** (2012). Haziran 2014 Top500 listesinin ilk onundaki iki süperbilgisayarın yapı taşıdır.
    - 61 x86 çekirdeği vardır; bunlar yoğun biçimde özelleştirilmiş Pentium çekirdekleridir.
    - Her çekirdek boru hattı duraklamalarını ve bellek gecikmelerini gizlemek için 4 thread'i aynı anda yürütür. 512 bitlik vektör birimi (VPU) saat başına 16 tek ya da 8 çift duyarlıklı işlem yapar.
    - Çekirdekler, her yönde üçer halkadan oluşan çift yönlü bir halkayla bağlıdır: 64 bayt genişliğinde bir veri halkası, adres (AD) ve onay (AK) halkaları. Tutarlılığı dağıtık etiket dizinleri yönetir.
    - PCIe kartı olarak Linux çalıştırır ve ana makineye ağdaki başka bir makine gibi görünür: SSH ile bağlanılır. Uygulamalar ana makinede çalışıp bir kısmını karta aktarabilir ya da tamamen kartta (yerel kipte) çalışabilir. Thread'ler, OpenMP, TBB, MPI gibi var olan araçlar kullanılır; GPU'ların yeni araç ve teknik öğrenmeyi gerektirmesine karşı bu bir avantajdır.

Çok çekirdekli bütün mimarilerin ortak bir özelliği görece düşük saat frekansıdır: o dönemde GPU'lar 0.8–1.5 GHz, TILE-Gx 1.2 GHz, Xeon Phi 1.2–1.3 GHz. Bu, milyarlarca transistörü aynı kalıba sığdırmanın bedelidir; yukarıdaki güç hesabı nedenini gösterir.

:::uyarı Kaynak düzeltmeleri (lect_01 slayt 27–29 ve 36)
**Cell.** Slayt Cell'in 2007'de tanıtıldığını söylüyor; Cell 2005'te duyuruldu ve PS3 Kasım 2006'da piyasaya çıktı. PS3'teki yongada sekiz SPE bulunur, ama biri üretim verimi için kapatılmıştır: oyunlar yedisinden altısını kullanabilir. Slayttaki "sekiz SPE'nin toplam 102.4 GFLOPS çift duyarlıklı tepe başarımı" da özgün Cell'e değil, 2008'deki PowerXCell 8i'ye aittir. Özgün Cell'in SPE'leri tek duyarlıkta güçlü (sekiz SPE 3.2 GHz'te 204.8 GFLOPS), çift duyarlıkta zayıftı (yaklaşık 15 GFLOPS). Roadrunner'ın PowerXCell 8i kullanması bu yüzdendir.

**Kepler.** Slayt "çekirdek başına 255 adet 32 bitlik yazmaç" diyor. 255, bir **thread'in** kullanabileceği en çok yazmaç sayısıdır (GK110). Yazmaç dosyası SMX başına 65 536 adet 32 bitlik yazmaçtır ve çekirdeklere değil, o SMX'te yerleşik thread'lere paylaştırılır. Thread başına yazmaç arttıkça SMX'te aynı anda barınabilen thread sayısı (doluluk, occupancy) düşer (M10.4).
:::

### Başarım ölçüleri {#olculer}

Çok çekirdekli donanım ve yazılım çabalarının amacı daha fazla başarımdır: daha kısa yürütme süresi, daha büyük problemler ve veri kümeleri. Bu çabaların ne kadar etkili olduğunu değerlendirmek için nesnel ölçütler gerekir. Paralel bir program en azından yürütme süresinde sıralı karşılığını geçebilmelidir. Paralel kod geliştirmek ve hata ayıklamak çok daha zordur; kazancın çabaya değdiğinden emin olunmalıdır.

**Hızlanma.** Yürütme süresindeki iyileşme hızlanmayla ifade edilir:
$$\text{hızlanma}=\frac{t_{\text{sıralı}}}{t_{\text{paralel}}}.$$
Burada iki süre problemin aynı örneği için ölçülür. Süreleri birçok etken etkiler:

- programcının becerisi;
- derleyici (GNU ya da Intel C++) ve derleyici seçenekleri (eniyileme açık/kapalı);
- işletim sistemi;
- girdiyi tutan dosya sistemi (EXT4, NTFS);
- günün saati (farklı iş yükleri, ağ trafiği).

Bildirilen hızlanmalara güvenmek için iki **altın kural** vardır:

1. Sıralı ve paralel programlar özdeş yazılım ve donanımda, benzer koşullarda sınanmalıdır.
2. Sıralı program, problemin bilinen **en hızlı** çözümü olmalıdır. Paralel algoritmanın tek thread'de çalıştırılması sıralı karşılaştırma ölçütü değildir.

Belirli bir işlemci sayısında hızlanma girdiye göre de değişebilir. Bu yüzden aynı boyutta birçok girdi üzerinde ortalama, hatta ortalama, en büyük ve en küçük değer bildirilir.

**Verim.** Hızlanma, bir problemin hızlandırılmasının mümkün olup olmadığını söyler: hızlanma > 1 mi? Ama bunun ılımlı bir kaynakla, verimli biçimde yapılıp yapılamayacağını söylemez. İkinci ölçüt verimdir:
$$\text{verim}=\frac{\text{hızlanma}}{N}=\frac{t_{\text{sıralı}}}{N\,t_{\text{paralel}}}.$$
$N$, paralel yürütmede kullanılan CPU ya da çekirdek sayısıdır. Verim, paralel yürütme sırasında bir düğümün ortalama kullanım oranı olarak yorumlanabilir. %100 verim, hızlanmanın $N$ olması ve işin $N$ işlemciye eşit bölünüp hepsinin her an kullanılması demektir; buna **doğrusal hızlanma** denir. Doğrusal hızlanma ne yazık ki idealdir. Birden çok işlemci bir amaç için çalışırken mesaj alışverişiyle ya da paylaşılan kaynakları yöneterek birbirleriyle eşgüdüm sağlamak için zaman harcar. Bu etkinlik CPU zamanı çalar ve hızlanmayı $N$'nin altına düşürür.

### Amdahl yasası {#amdahl}

Gene Amdahl 1967'de paralel bir programdan beklenebilecek başarım için basit bir düşünce deneyi yaptı. Varsayımları şunlardır:

- Sıralı bir uygulama tek CPU'da $T$ sürede çalışır.
- Uygulamanın $0\le\alpha\le1$ oranındaki kısmı paralelleştirilebilir, kalan $1-\alpha$ sıralı yapılmalıdır.
- Paralel yürütmenin iletişim ek yükü yoktur ve paralel kısım istenen sayıda CPU'ya eşit bölünebilir.

Son varsayım, çekirdeklerin aynı paylaşımlı belleğe eriştiği çok çekirdekli mimarilere özellikle uyar. $N$ düğümle hızlanma
$$\text{hızlanma}=\frac{T}{(1-\alpha)T+\frac{\alpha T}N}=\frac1{(1-\alpha)+\frac\alpha N}\;\xrightarrow{N\to\infty}\;\frac1{1-\alpha}$$
ile sınırlıdır. Öngörülen hızlanma çarpıcı biçimde sınırlıdır: $\alpha=0.9$ ise işlemci sayısı ne olursa olsun hızlanma 10'un altında kalır. $\alpha=0.9$, $N=8$ için hızlanma 4.71, verim 0.59'dur.

Aşağıdaki deneyde $f$, **seri** kesirdir ($f=1-\alpha$): $f=0.05$ ile 1024 çekirdekte hızlanma 19.6, tavan 20. İletişim ek yükü eklenince en iyi bir $N$ belirir; o noktadan sonra işlemci eklemek programı yavaşlatır.

::lab l1

**Karıncalar ve filler.** Amdahl yasasının ilginç bir sonucu vardır ve ortaya çıkış nedeni bu olabilir. Yasa, ana bilgisayarlardan çok daha ucuz ama daha az güçlü mini bilgisayarların çıktığı bir dönemde yazıldı. Doğal soru şuydu: bir problemi hızlandırmak için en iyi yatırım hangisi, birkaç güçlü ve pahalı makine mi, yoksa çok sayıda daha az güçlü ama ucuz makine mi? Bu ikilemi anlatan benzetme sonucu baştan ima ediyordu: "karınca ordusuna karşı fil sürüsü".

Bir program güçlü CPU A'da $T_A$, ucuz CPU B'de $T_B$ sürede çalışsın; A, $r=T_B/T_A$ kat hızlıdır. $N_B$ tane B ile tek A'ya göre en iyi hızlanma
$$\text{hızlanma}=\frac{T_A}{(1-\alpha)T_B+\frac{\alpha T_B}{N_B}}=\frac1{(1-\alpha)r+\frac{\alpha r}{N_B}}\;\xrightarrow{N_B\to\infty}\;\frac1{(1-\alpha)r}.$$
$r\ge1/(1-\alpha)$ ise, yani zayıf işlemci bu kadar yavaşsa, kaç "karınca" kullanılırsa kullanılsın hızlanma 1'i geçmez. $\alpha=0.9$ ve $r=10$ ise tek bir pahalı CPU paralel yoldan iyidir.

<figure class="fig wide"><svg viewBox="0 0 640 236" role="img" aria-label="Amdahl çerçevesinde çok sayıda yavaş işlemci ile tek hızlı işlemcinin karşılaştırılması: hızlanma eğrileri ve tavanları"><path d="M60,200.0 L390,200.0" class="mute"/><text x="54" y="204.0" text-anchor="end" class="sm">0.1</text><path d="M60,137.1 L390,137.1" class="mute"/><text x="54" y="141.1" text-anchor="end" class="sm">0.5</text><path d="M60,110.0 L390,110.0" class="mute"/><text x="54" y="114.0" text-anchor="end" class="sm">1</text><path d="M60,82.9 L390,82.9" class="mute"/><text x="54" y="86.9" text-anchor="end" class="sm">2</text><path d="M60,47.1 L390,47.1" class="mute"/><text x="54" y="51.1" text-anchor="end" class="sm">5</text><path d="M60,20.0 L390,20.0" class="mute"/><text x="54" y="24.0" text-anchor="end" class="sm">10</text><text x="60.0" y="216" text-anchor="middle" class="sm">1</text><text x="142.5" y="216" text-anchor="middle" class="sm">10</text><text x="225.0" y="216" text-anchor="middle" class="sm">100</text><text x="307.5" y="216" text-anchor="middle" class="sm">1000</text><text x="390.0" y="216" text-anchor="middle" class="sm">10000</text><text x="225.0" y="232" text-anchor="middle" class="sm">ucuz işlemci sayısı N<tspan baseline-shift="-3" font-size="75%">B</tspan></text><path d="M60.0,137.1 L61.7,135.5 L63.3,133.9 L65.0,132.2 L66.6,130.6 L68.3,129.1 L69.9,127.5 L71.6,125.9 L73.3,124.3 L74.9,122.8 L76.6,121.2 L78.2,119.7 L79.9,118.2 L81.6,116.7 L83.2,115.2 L84.9,113.7 L86.5,112.2 L88.2,110.8 L89.8,109.3 L91.5,107.9 L93.2,106.5 L94.8,105.0 L96.5,103.7 L98.1,102.3 L99.8,100.9 L101.5,99.6 L103.1,98.2 L104.8,96.9 L106.4,95.6 L108.1,94.4 L109.7,93.1 L111.4,91.9 L113.1,90.6 L114.7,89.4 L116.4,88.2 L118.0,87.1 L119.7,85.9 L121.4,84.8 L123.0,83.7 L124.7,82.6 L126.3,81.5 L128.0,80.5 L129.6,79.4 L131.3,78.4 L133.0,77.5 L134.6,76.5 L136.3,75.5 L137.9,74.6 L139.6,73.7 L141.3,72.8 L142.9,72.0 L144.6,71.1 L146.2,70.3 L147.9,69.5 L149.5,68.7 L151.2,68.0 L152.9,67.2 L154.5,66.5 L156.2,65.8 L157.8,65.1 L159.5,64.5 L161.2,63.8 L162.8,63.2 L164.5,62.6 L166.1,62.0 L167.8,61.5 L169.4,60.9 L171.1,60.4 L172.8,59.9 L174.4,59.4 L176.1,58.9 L177.7,58.4 L179.4,58.0 L181.1,57.6 L182.7,57.1 L184.4,56.7 L186.0,56.3 L187.7,56.0 L189.3,55.6 L191.0,55.3 L192.7,54.9 L194.3,54.6 L196.0,54.3 L197.6,54.0 L199.3,53.7 L201.0,53.4 L202.6,53.2 L204.3,52.9 L205.9,52.7 L207.6,52.4 L209.2,52.2 L210.9,52.0 L212.6,51.8 L214.2,51.6 L215.9,51.4 L217.5,51.2 L219.2,51.0 L220.9,50.9 L222.5,50.7 L224.2,50.5 L225.8,50.4 L227.5,50.2 L229.1,50.1 L230.8,50.0 L232.5,49.8 L234.1,49.7 L235.8,49.6 L237.4,49.5 L239.1,49.4 L240.8,49.3 L242.4,49.2 L244.1,49.1 L245.7,49.0 L247.4,48.9 L249.0,48.9 L250.7,48.8 L252.4,48.7 L254.0,48.6 L255.7,48.6 L257.3,48.5 L259.0,48.4 L260.7,48.4 L262.3,48.3 L264.0,48.3 L265.6,48.2 L267.3,48.2 L268.9,48.1 L270.6,48.1 L272.3,48.0 L273.9,48.0 L275.6,47.9 L277.2,47.9 L278.9,47.9 L280.6,47.8 L282.2,47.8 L283.9,47.8 L285.5,47.7 L287.2,47.7 L288.8,47.7 L290.5,47.7 L292.2,47.6 L293.8,47.6 L295.5,47.6 L297.1,47.6 L298.8,47.5 L300.5,47.5 L302.1,47.5 L303.8,47.5 L305.4,47.5 L307.1,47.4 L308.7,47.4 L310.4,47.4 L312.1,47.4 L313.7,47.4 L315.4,47.4 L317.0,47.4 L318.7,47.3 L320.4,47.3 L322.0,47.3 L323.7,47.3 L325.3,47.3 L327.0,47.3 L328.6,47.3 L330.3,47.3 L332.0,47.3 L333.6,47.3 L335.3,47.3 L336.9,47.2 L338.6,47.2 L340.3,47.2 L341.9,47.2 L343.6,47.2 L345.2,47.2 L346.9,47.2 L348.5,47.2 L350.2,47.2 L351.9,47.2 L353.5,47.2 L355.2,47.2 L356.8,47.2 L358.5,47.2 L360.2,47.2 L361.8,47.2 L363.5,47.2 L365.1,47.2 L366.8,47.2 L368.4,47.2 L370.1,47.2 L371.8,47.2 L373.4,47.1 L375.1,47.1 L376.7,47.1 L378.4,47.1 L380.1,47.1 L381.7,47.1 L383.4,47.1 L385.0,47.1 L386.7,47.1 L388.3,47.1 L390.0,47.1" class="cp w2" fill="none"/><path d="M60.0,172.9 L61.7,171.3 L63.3,169.7 L65.0,168.1 L66.6,166.5 L68.3,164.9 L69.9,163.3 L71.6,161.7 L73.3,160.1 L74.9,158.6 L76.6,157.1 L78.2,155.5 L79.9,154.0 L81.6,152.5 L83.2,151.0 L84.9,149.5 L86.5,148.0 L88.2,146.6 L89.8,145.1 L91.5,143.7 L93.2,142.3 L94.8,140.9 L96.5,139.5 L98.1,138.1 L99.8,136.7 L101.5,135.4 L103.1,134.1 L104.8,132.8 L106.4,131.5 L108.1,130.2 L109.7,128.9 L111.4,127.7 L113.1,126.5 L114.7,125.2 L116.4,124.1 L118.0,122.9 L119.7,121.7 L121.4,120.6 L123.0,119.5 L124.7,118.4 L126.3,117.3 L128.0,116.3 L129.6,115.3 L131.3,114.3 L133.0,113.3 L134.6,112.3 L136.3,111.4 L137.9,110.4 L139.6,109.5 L141.3,108.6 L142.9,107.8 L144.6,106.9 L146.2,106.1 L147.9,105.3 L149.5,104.5 L151.2,103.8 L152.9,103.0 L154.5,102.3 L156.2,101.6 L157.8,100.9 L159.5,100.3 L161.2,99.6 L162.8,99.0 L164.5,98.4 L166.1,97.8 L167.8,97.3 L169.4,96.7 L171.1,96.2 L172.8,95.7 L174.4,95.2 L176.1,94.7 L177.7,94.2 L179.4,93.8 L181.1,93.4 L182.7,92.9 L184.4,92.5 L186.0,92.2 L187.7,91.8 L189.3,91.4 L191.0,91.1 L192.7,90.7 L194.3,90.4 L196.0,90.1 L197.6,89.8 L199.3,89.5 L201.0,89.2 L202.6,89.0 L204.3,88.7 L205.9,88.5 L207.6,88.2 L209.2,88.0 L210.9,87.8 L212.6,87.6 L214.2,87.4 L215.9,87.2 L217.5,87.0 L219.2,86.8 L220.9,86.7 L222.5,86.5 L224.2,86.4 L225.8,86.2 L227.5,86.1 L229.1,85.9 L230.8,85.8 L232.5,85.7 L234.1,85.5 L235.8,85.4 L237.4,85.3 L239.1,85.2 L240.8,85.1 L242.4,85.0 L244.1,84.9 L245.7,84.8 L247.4,84.7 L249.0,84.7 L250.7,84.6 L252.4,84.5 L254.0,84.4 L255.7,84.4 L257.3,84.3 L259.0,84.2 L260.7,84.2 L262.3,84.1 L264.0,84.1 L265.6,84.0 L267.3,84.0 L268.9,83.9 L270.6,83.9 L272.3,83.8 L273.9,83.8 L275.6,83.8 L277.2,83.7 L278.9,83.7 L280.6,83.6 L282.2,83.6 L283.9,83.6 L285.5,83.6 L287.2,83.5 L288.8,83.5 L290.5,83.5 L292.2,83.4 L293.8,83.4 L295.5,83.4 L297.1,83.4 L298.8,83.4 L300.5,83.3 L302.1,83.3 L303.8,83.3 L305.4,83.3 L307.1,83.3 L308.7,83.2 L310.4,83.2 L312.1,83.2 L313.7,83.2 L315.4,83.2 L317.0,83.2 L318.7,83.2 L320.4,83.2 L322.0,83.1 L323.7,83.1 L325.3,83.1 L327.0,83.1 L328.6,83.1 L330.3,83.1 L332.0,83.1 L333.6,83.1 L335.3,83.1 L336.9,83.1 L338.6,83.1 L340.3,83.0 L341.9,83.0 L343.6,83.0 L345.2,83.0 L346.9,83.0 L348.5,83.0 L350.2,83.0 L351.9,83.0 L353.5,83.0 L355.2,83.0 L356.8,83.0 L358.5,83.0 L360.2,83.0 L361.8,83.0 L363.5,83.0 L365.1,83.0 L366.8,83.0 L368.4,83.0 L370.1,83.0 L371.8,83.0 L373.4,83.0 L375.1,83.0 L376.7,83.0 L378.4,83.0 L380.1,83.0 L381.7,83.0 L383.4,82.9 L385.0,82.9 L386.7,82.9 L388.3,82.9 L390.0,82.9" class="ca w2" fill="none"/><path d="M60.0,200.0 L61.7,198.4 L63.3,196.8 L65.0,195.2 L66.6,193.6 L68.3,192.0 L69.9,190.4 L71.6,188.8 L73.3,187.2 L74.9,185.7 L76.6,184.1 L78.2,182.6 L79.9,181.1 L81.6,179.6 L83.2,178.1 L84.9,176.6 L86.5,175.1 L88.2,173.7 L89.8,172.2 L91.5,170.8 L93.2,169.4 L94.8,168.0 L96.5,166.6 L98.1,165.2 L99.8,163.8 L101.5,162.5 L103.1,161.2 L104.8,159.8 L106.4,158.5 L108.1,157.3 L109.7,156.0 L111.4,154.8 L113.1,153.5 L114.7,152.3 L116.4,151.1 L118.0,150.0 L119.7,148.8 L121.4,147.7 L123.0,146.6 L124.7,145.5 L126.3,144.4 L128.0,143.4 L129.6,142.4 L131.3,141.3 L133.0,140.4 L134.6,139.4 L136.3,138.4 L137.9,137.5 L139.6,136.6 L141.3,135.7 L142.9,134.9 L144.6,134.0 L146.2,133.2 L147.9,132.4 L149.5,131.6 L151.2,130.9 L152.9,130.1 L154.5,129.4 L156.2,128.7 L157.8,128.0 L159.5,127.4 L161.2,126.7 L162.8,126.1 L164.5,125.5 L166.1,124.9 L167.8,124.4 L169.4,123.8 L171.1,123.3 L172.8,122.8 L174.4,122.3 L176.1,121.8 L177.7,121.3 L179.4,120.9 L181.1,120.5 L182.7,120.0 L184.4,119.6 L186.0,119.3 L187.7,118.9 L189.3,118.5 L191.0,118.2 L192.7,117.8 L194.3,117.5 L196.0,117.2 L197.6,116.9 L199.3,116.6 L201.0,116.3 L202.6,116.1 L204.3,115.8 L205.9,115.6 L207.6,115.3 L209.2,115.1 L210.9,114.9 L212.6,114.7 L214.2,114.5 L215.9,114.3 L217.5,114.1 L219.2,113.9 L220.9,113.8 L222.5,113.6 L224.2,113.4 L225.8,113.3 L227.5,113.2 L229.1,113.0 L230.8,112.9 L232.5,112.8 L234.1,112.6 L235.8,112.5 L237.4,112.4 L239.1,112.3 L240.8,112.2 L242.4,112.1 L244.1,112.0 L245.7,111.9 L247.4,111.8 L249.0,111.8 L250.7,111.7 L252.4,111.6 L254.0,111.5 L255.7,111.5 L257.3,111.4 L259.0,111.3 L260.7,111.3 L262.3,111.2 L264.0,111.2 L265.6,111.1 L267.3,111.1 L268.9,111.0 L270.6,111.0 L272.3,110.9 L273.9,110.9 L275.6,110.8 L277.2,110.8 L278.9,110.8 L280.6,110.7 L282.2,110.7 L283.9,110.7 L285.5,110.6 L287.2,110.6 L288.8,110.6 L290.5,110.6 L292.2,110.5 L293.8,110.5 L295.5,110.5 L297.1,110.5 L298.8,110.4 L300.5,110.4 L302.1,110.4 L303.8,110.4 L305.4,110.4 L307.1,110.4 L308.7,110.3 L310.4,110.3 L312.1,110.3 L313.7,110.3 L315.4,110.3 L317.0,110.3 L318.7,110.3 L320.4,110.2 L322.0,110.2 L323.7,110.2 L325.3,110.2 L327.0,110.2 L328.6,110.2 L330.3,110.2 L332.0,110.2 L333.6,110.2 L335.3,110.2 L336.9,110.2 L338.6,110.1 L340.3,110.1 L341.9,110.1 L343.6,110.1 L345.2,110.1 L346.9,110.1 L348.5,110.1 L350.2,110.1 L351.9,110.1 L353.5,110.1 L355.2,110.1 L356.8,110.1 L358.5,110.1 L360.2,110.1 L361.8,110.1 L363.5,110.1 L365.1,110.1 L366.8,110.1 L368.4,110.1 L370.1,110.1 L371.8,110.1 L373.4,110.1 L375.1,110.1 L376.7,110.1 L378.4,110.0 L380.1,110.0 L381.7,110.0 L383.4,110.0 L385.0,110.0 L386.7,110.0 L388.3,110.0 L390.0,110.0" class="cb w2" fill="none"/><path d="M60.0,227.1 L61.7,225.5 L63.3,223.9 L65.0,222.2 L66.6,220.6 L68.3,219.1 L69.9,217.5 L71.6,215.9 L73.3,214.3 L74.9,212.8 L76.6,211.2 L78.2,209.7 L79.9,208.2 L81.6,206.7 L83.2,205.2 L84.9,203.7 L86.5,202.2 L88.2,200.8 L89.8,199.3 L91.5,197.9 L93.2,196.5 L94.8,195.0 L96.5,193.7 L98.1,192.3 L99.8,190.9 L101.5,189.6 L103.1,188.2 L104.8,186.9 L106.4,185.6 L108.1,184.4 L109.7,183.1 L111.4,181.9 L113.1,180.6 L114.7,179.4 L116.4,178.2 L118.0,177.1 L119.7,175.9 L121.4,174.8 L123.0,173.7 L124.7,172.6 L126.3,171.5 L128.0,170.5 L129.6,169.4 L131.3,168.4 L133.0,167.5 L134.6,166.5 L136.3,165.5 L137.9,164.6 L139.6,163.7 L141.3,162.8 L142.9,162.0 L144.6,161.1 L146.2,160.3 L147.9,159.5 L149.5,158.7 L151.2,158.0 L152.9,157.2 L154.5,156.5 L156.2,155.8 L157.8,155.1 L159.5,154.5 L161.2,153.8 L162.8,153.2 L164.5,152.6 L166.1,152.0 L167.8,151.5 L169.4,150.9 L171.1,150.4 L172.8,149.9 L174.4,149.4 L176.1,148.9 L177.7,148.4 L179.4,148.0 L181.1,147.6 L182.7,147.1 L184.4,146.7 L186.0,146.3 L187.7,146.0 L189.3,145.6 L191.0,145.3 L192.7,144.9 L194.3,144.6 L196.0,144.3 L197.6,144.0 L199.3,143.7 L201.0,143.4 L202.6,143.2 L204.3,142.9 L205.9,142.7 L207.6,142.4 L209.2,142.2 L210.9,142.0 L212.6,141.8 L214.2,141.6 L215.9,141.4 L217.5,141.2 L219.2,141.0 L220.9,140.9 L222.5,140.7 L224.2,140.5 L225.8,140.4 L227.5,140.2 L229.1,140.1 L230.8,140.0 L232.5,139.8 L234.1,139.7 L235.8,139.6 L237.4,139.5 L239.1,139.4 L240.8,139.3 L242.4,139.2 L244.1,139.1 L245.7,139.0 L247.4,138.9 L249.0,138.9 L250.7,138.8 L252.4,138.7 L254.0,138.6 L255.7,138.6 L257.3,138.5 L259.0,138.4 L260.7,138.4 L262.3,138.3 L264.0,138.3 L265.6,138.2 L267.3,138.2 L268.9,138.1 L270.6,138.1 L272.3,138.0 L273.9,138.0 L275.6,137.9 L277.2,137.9 L278.9,137.9 L280.6,137.8 L282.2,137.8 L283.9,137.8 L285.5,137.7 L287.2,137.7 L288.8,137.7 L290.5,137.7 L292.2,137.6 L293.8,137.6 L295.5,137.6 L297.1,137.6 L298.8,137.5 L300.5,137.5 L302.1,137.5 L303.8,137.5 L305.4,137.5 L307.1,137.4 L308.7,137.4 L310.4,137.4 L312.1,137.4 L313.7,137.4 L315.4,137.4 L317.0,137.4 L318.7,137.3 L320.4,137.3 L322.0,137.3 L323.7,137.3 L325.3,137.3 L327.0,137.3 L328.6,137.3 L330.3,137.3 L332.0,137.3 L333.6,137.3 L335.3,137.3 L336.9,137.2 L338.6,137.2 L340.3,137.2 L341.9,137.2 L343.6,137.2 L345.2,137.2 L346.9,137.2 L348.5,137.2 L350.2,137.2 L351.9,137.2 L353.5,137.2 L355.2,137.2 L356.8,137.2 L358.5,137.2 L360.2,137.2 L361.8,137.2 L363.5,137.2 L365.1,137.2 L366.8,137.2 L368.4,137.2 L370.1,137.2 L371.8,137.2 L373.4,137.1 L375.1,137.1 L376.7,137.1 L378.4,137.1 L380.1,137.1 L381.7,137.1 L383.4,137.1 L385.0,137.1 L386.7,137.1 L388.3,137.1 L390.0,137.1" class="cq w2" fill="none"/><path d="M60,110.0 L390,110.0" class="mute w2" stroke-dasharray="4 3"/><text x="405" y="34" class="sm">α = 0.9; hızlı işlemci A, r kat yavaş B</text><text x="405" y="52" class="sm">S = 1 / ((1 − α) r + α r / N<tspan baseline-shift="-3" font-size="75%">B</tspan>)</text><text x="405" y="80" class="tp">r = 2: tavan 1/((1 − α) r) = 5</text><text x="405" y="100" class="ta">r = 5: tavan 1/((1 − α) r) = 2</text><text x="405" y="120" class="tb">r = 10: tavan 1/((1 − α) r) = 1</text><text x="405" y="140" class="tq">r = 20: tavan 1/((1 − α) r) = 0.5</text><text x="405" y="172" class="sm">r ≥ 1/(1 − α) = 10 ise sayısız</text><text x="405" y="190" class="sm">"karınca" bir "fili" geçemez.</text></svg><figcaption>Şekil 2 — Amdahl'ın varsayımlarıyla (sabit problem, sıfır iletişim) yavaş işlemciler ancak r < 1/(1 − α) ise tek hızlı işlemciyi geçebilir; kesikli çizgi hızlanma 1 (hesaplandı, eksenler logaritmik).</figcaption></figure>

Ancak bu sonuç yüksek başarımlı hesabın gerçeğiyle uyuşmaz. Dünyanın en güçlü 500 süperbilgisayarının listesi yalnız, masaüstü makinelerde bulunabilecek hazır bileşenlerden kurulmuş, on binlerce ile milyonlarca çekirdekli makinelerle doludur. Karınca ordusu kazandıysa yaklaşımda bir kusur olmalıdır.

### Gustafson–Barsis {#gustafson}

Amdahl'dan yirmi yıl sonra Gustafson ve Barsis (1988) problemi farklı bir açıdan ele aldı: paralel bir platform sıralı bir programı hızlandırmaktan fazlasını yapar, **daha büyük problem örneklerini** barındırabilir. Paralel programın sıralıya göre ne yapabildiğine bakmak yerine, sıralı bir makinenin paralel makinenin çözebildiği problemi çözmesi gerekseydi nasıl davranacağına bakılmalıdır. Varsayımlar şunlardır:

- Paralel bir uygulama $N$ CPU'da $T$ sürede çalışır.
- Uygulama toplam sürenin $0\le\alpha\le1$ **oranında** bütün makinelerde, kalan $1-\alpha$ oranında sıralı çalışır.
- Paralel durum için toplam süre $T$ sabittir. İşlemci sayısı arttıkça yapılan toplam iş de artmalıdır.

Aynı problemi sıralı bir makinede çözmek, paralel kısımların da sırayla yapılmasını gerektirir: $t_{\text{sıralı}}=(1-\alpha)T+N\alpha T$. Buradan
$$\text{hızlanma}=\frac{(1-\alpha)T+N\alpha T}T=(1-\alpha)+N\alpha,\qquad\text{verim}=\frac{1-\alpha}N+\alpha.$$
Ölçeklenmiş hızlanma $N$ ile **doğrusal** büyür: $\alpha=0.9$, $N=8$ ile 7.3.

Gustafson–Barsis'in de eksikleri vardır: iletişim ek yükünü sıfır kabul eder. Bu öngörüler gerçek olamayacak kadar iyidir. "Utanç verici derecede paralel" (embarrassingly parallel) denen problemlerde bile $N$ büyüdükçe iletişim ek yükü belirleyici olur, hızlanma kazançları azalır ve verim çöker. Pratikte %90'ın üzerinde verim elde etmek kayda değer bir başarı sayılır.

:::uyarı Kaynak düzeltmesi (lect_01 slayt 66): Amdahl yasası "temelden kusurlu" değildir
Slayt Amdahl yasasının "temelden kusurlu" olduğunu ve paralel programların öngörülen sınırları "rutin olarak aştığını" söylüyor. İki yasa farklı soruları yanıtlar:

- **Amdahl** problem boyu sabitken (**güçlü ölçekleme**) doğrudur. Sabit bir işi daha çok işlemciyle bitirmek istediğimizde seri kısım gerçekten tavandır.
- **Gustafson** problem işlemci sayısıyla büyürken (**zayıf ölçekleme**) doğrudur.

İki formül aynı programı farklı $\alpha$'larla betimler. Amdahl'ın $\alpha$'sı tek işlemcideki süre oranıdır. Gustafson'unki $N$ işlemcideki süre oranıdır ve problem büyüdükçe artar. Sınırın "aşılması" ise genellikle **süper doğrusal** hızlanmadan gelir: problem $N$ parçaya bölününce her parça önbelleğe sığar ve parça başına iş sıralı programdakinden hızlı yapılır. Bu Amdahl'ın varsayımını (paralel kısım tam $N$ kat hızlanır) bozar, yasayı değil. Top500 makinelerinin başarısı da zayıf ölçeklemenin başarısıdır: büyük makineler daha büyük simülasyonları aynı sürede çözer.
:::

**Karp–Flatt ölçüsü.** Ölçülen hızlanmadan seri kesri geri çözmek kullanışlıdır. Amdahl formülünü $e=1-\alpha$ için çözersek
$$e=\frac{1/S-1/p}{1-1/p}$$
olur (Karp ve Flatt 1990). $S=4.70588$ ve $p=8$ ise $e=0.1$. $e$, $p$ arttıkça sabit kalıyorsa sınırlayıcı olan gerçekten seri koddur. **Artıyorsa** hızlanmayı yiyen, işlemci sayısıyla büyüyen bir ek yüktür: iletişim, eşgüdüm ya da yük dengesizliği.

**Gecikme ve verim.** Ders bu bölümü iki işlemci türünü karşılaştıran bir çizimle kapatıyor. Gecikmeye eniyilenmiş işlemci bir iş isteğine hızla yanıt verir ama birim zamanda az iş çıkarır. Verime eniyilenmiş işlemci ilk sonucu geç verir, ama sonra birim zamanda çok daha fazla çıktı üretir. Hangisinin iyi olduğu işin yapısına bağlıdır. Etkileşimli bir istek için gecikme, toplu bir hesap için verim önemlidir.

### İş–yol modeli {#is-yol}

Amdahl modeli programı iki parçaya (seri ve paralel) ayırır. Gerçek programlarda paralellik bir **görev bağımlılık grafiğiyle** (DAG) betimlenir: düğümler görevler, kenarlar "şu bitmeden bu başlayamaz" ilişkileridir. İki sayı grafiği özetler:

- **İş** $W$: bütün görevlerin toplam maliyeti, yani tek işlemcide süre $T_1$.
- **Yol** $S_\infty$ (span, kritik yol): grafikteki en uzun ağırlıklı yol, yani sonsuz işlemcide süre $T_\infty$.

Oranları $W/S_\infty$ **paralelliktir**: yararlı biçimde kullanılabilecek işlemci sayısının üst sınırı.

:::türetim Açgözlü zamanlamanın sınırları (Graham 1966, Brent 1974)
Alt sınırlar açıktır. $p$ işlemci zaman birimi başına en çok $p$ birim iş yapar, dolayısıyla $T_p\ge W/p$. Kritik yoldaki görevler sırayla yapılmalıdır, dolayısıyla $T_p\ge S_\infty$.

**Açgözlü** bir zamanlamada hazır bir görev varken hiçbir işlemci boş durmaz. Böyle bir zamanlamada her zaman adımı iki türden birindedir:

- **Tam adım:** $p$ işlemcinin hepsi çalışır. Bu tür adımlar toplam en çok $W/p$ tane olabilir, çünkü her biri $p$ birim iş tüketir.
- **Eksik adım:** bazı işlemciler boştur. Bu, o anda hazır olan **bütün** görevlerin çalıştığı anlamına gelir. Kalan grafiğin en uzun yolunun başındaki görev de hazırdır ve çalışmaktadır; dolayısıyla her eksik adım kalan kritik yolu bir birim kısaltır. Eksik adımlar toplam en çok $S_\infty$ tane olabilir.

Toplarsak
$$\max\Bigl(\frac Wp,S_\infty\Bigr)\le T_p\le\frac Wp+S_\infty$$
elde edilir. Açgözlü her zamanlama en iyisinin iki katı içindedir, çünkü üst sınır alt sınırın en çok iki katıdır. $p\ll W/S_\infty$ ise $T_p\approx W/p$, yani doğrusal hızlanma; $p\gg W/S_\infty$ ise $T_p\approx S_\infty$ ve ek işlemciler boşta kalır.
:::

Örnekler:

- **Ağaç indirgeme.** 1024 sayının ağaçla toplanması $W=1023$ toplama ve $S_\infty=\log_21024=10$ adım ister; paralellik yaklaşık 100'dür.
- **Merge sort.** Birleştirme adımı seri yazıldığında $W=n\log_2n+n$ olur. Yol ise son birleştirmelerin toplamıdır, $n+n/2+\dots\approx2n$. Paralellik yalnız $(\log_2n+1)/2$'dir: $n=128$ için 4.

Aşağıdaki deney bu grafiklerde açgözlü liste zamanlamayı çalıştırıyor. Merge sort'ta 8 işlemciyle hızlanma 3.37'de kalıyor (verim %42). Ağaç indirgemede aynı işlemci sayısı %88 verim veriyor. Düzensiz bir DAG'da hazır kuyruğunu **kritik yol önceliğiyle** sıralamak, FIFO'nun 94 birimlik süresini 86'ya, yani neredeyse alt sınıra (85.75) indiriyor.

::lab l2

### Paralel program tasarımı: PCAM {#pcam}

Çok çekirdekli programlamaya geçiş deneyimli programcılar için bile kolay değildir. Paralel programlama, deyimlerin sıkı bir sırayla yürüdüğü sıralı programın mirasını bozar. Birçok şey aynı anda olduğunda deyimlerin olası sıralaması tek değildir. Program dikkatle tasarlanmazsa verinin hatalı güncellenmesi ya da programı kilitleyen eşzamansız iletişim gibi sorunlar sızar. Tasarımın iki amacı vardır: **doğruluk** ve **başarım**.

İki hatırlatma yapmak gerekir. Sıralı bir programı paralele çevirmek her zaman doğru yol değildir. Paralel gerçekleştirimin gerekli ve ulaşılabilir olduğundan emin olunmalıdır: algoritma paralel gerçekleştirime izin vermeyebilir ya da gerekli kaynaklar (zaman, mühendislik gücü) olmayabilir. Paralel yürütme tamamen yeni bir algoritma da gerektirebilir. GPU'lar, potansiyellerinden yararlanmak için program yapısına o kadar çok kısıt koyar ki genellikle baştan tasarım gerekir.

**PCAM**, Ian Foster'ın 1995 tarihli kitabıyla yaygınlaşan dört adımlı bir tasarım yöntemidir: bölümleme (Partitioning), iletişim (Communication), birleştirme (Agglomeration) ve eşleme (Mapping). Kabaca, eşzamanlı yapılabilecek iş parçalarını bul ve iletişim gereksinimlerini göz önünde tutarak işlemcilere ata. Foster'ın kitabı bugün de yöntemin kesin kaynağıdır ve fikirleri çok çekirdekli platformlar için hâlâ geçerlidir.

1. **Bölümleme.** Hesap olabildiğince çok bağımsız parçaya ayrılır; bu adım algoritmadaki paralelliği (varsa) ortaya çıkarır. Parçaların ayrıntı düzeyi (granularity) uygulamaya özgüdür. Ayrıştırma iki türlü olabilir:
    - **İşlev yönelimli:** gerçekleşen farklı adımları ayırmak (işlevsel ayrıştırma).
    - **Veri yönelimli:** işlenecek veriyi bölmek (alan ya da veri ayrıştırması).

    Pratik bir kural olarak parça sayısı, hesap düğümü sayısından bir–iki mertebe büyük olmalıdır. Bu, sonraki adımlara esneklik bırakır.
2. **İletişim.** İdealde parçalar tamamen bağımsızdır, ama genellikle bağımlılıklar vardır: biri başlamadan öteki bitmelidir. Bu bağımlılık veri aktarımını, yani iletişimi içerebilir. Bu adımda görevler arasında taşınacak veri hacmi belirlenir. İlk iki adım birlikte **görev bağımlılık grafiğini** verir: düğümler görevler, kenarlar iletişim hacmidir.
3. **Birleştirme.** İletişim paralel hesabı ciddi biçimde engelleyebilir. Onu yok etmenin bir yolu görevleri gruplamaktır: her grup tek bir düğüme atanacağı için grup içi iletişim ortadan kalkar. Pratik kural olarak grup sayısı düğüm sayısından bir mertebe büyük olmalıdır.
4. **Eşleme.** Gruplar düğümlere atanır. İki amaç vardır:
    - **Yük dengesi:** düğümlerin hepsine, yürütme süresiyle ölçüldüğünde, aşağı yukarı aynı iş düşmelidir.
    - **İletişimin daha da azaltılması:** aralarında pahalı veri alışverişi olan gruplar aynı düğüme konur. Paylaşımlı bellek üzerinden iletişim neredeyse bedavadır.

Adımların her birinin nasıl yapılacağı uygulamaya bağlıdır. Eşleme adımı genel biçimiyle NP-zordur (karar sürümü NP-tamdır): basit olmayan grafiklerde en iyisi bulunamaz, sezgisel yöntemler kullanılır. Başarım hesapları **çoğaltarak** da artırılabilir: bir değeri iki düğümde ayrı ayrı hesaplamak, sonucu iletmekten ucuz olabilir.

**Örnekler.**

- **Matris toplama** $C=A+B$: her eleman bağımsız bir görevdir ve iletişim yoktur. Birleştirme, satır ya da blok gruplarıdır; eşleme, eşit sayıda satırdır.
- **2-B evrişim:** her çıktı pikseli bir görevdir, ama komşu girdi piksellerine ihtiyaç duyar. Örneğin Laplace çekirdeği $\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$ ile bir çıktı için 5 girdi gerekir. Birleştirme görüntüyü bloklara ayırır. Bir bloğun yalnız **kenarındaki** pikseller komşu bloğun verisini (bir "hale", halo) ister. $b\times b$'lik bir blokta hesap $b^2$, iletişim $4b$ ile orantılıdır. Büyük bloklar iletişim/hesap oranını $4/b$'ye indirir.

### Ayrıştırma desenleri {#desenler}

Tasarım sürecinin en zor ve en kritik kısmı ayrıştırmadır: hesabın hangi kısımlarının eşzamanlı ilerleyebileceğini belirlemek. Görev grafiği en genel yaklaşımdır, ama geliştiricinin önceki deneyimden yararlanmasını sağlamaz. Burada **desenler** devreye girer. Mattson, Sanders ve Massingill (2004) bir iş yükünün paralel bir platformun düğümlerine dağıtılmak üzere ayrıştırılmasının temel yollarını kapsayan altı desen listeler; kitapta bunlara "algoritma yapısı tasarım uzayı" desenleri denir.

Desenler bir ağaç oluşturur. Üç ayrıştırma türünün her biri ikiye ayrılır:

- **İşlevsel ayrıştırma:** doğrusal ise görev paralelliği, özyinelemeli ise böl ve yönet.
- **Alan (veri) ayrıştırması:** doğrusal ise geometrik ayrıştırma, özyinelemeli ise özyinelemeli veri ayrıştırması.
- **Veri akışı ayrıştırması:** düzenli ise boru hattı, düzensiz ise olay tabanlı koordinasyon.

Veri akışı ayrıştırması, verinin bir dizi işlem aşamasından geçtiği akışlar için uygundur ve işlevsel ayrıştırmanın bir uzantısı sayılabilir. Desenler arasında katı bir ayrım yoktur: bir problem farklı biçimlerde ayrıştırılabilir ve desen çeşitliliği problemi farklı açılardan düşünmeye yardım eder.

- **Görev paralelliği.** Birçok sıralı kod, bir problemi birlikte çözen modüllerden oluşur. Örnek bir oyunun ana döngüsüdür: kullanıcı girdisini oku, ekranı çiz, sesleri çal, strateji hesapla. Her modül ayrı bir düğüme atanır ve bir **bariyerle** eşgüdüm sağlanır. Bariyer, bütün thread'ler o noktaya ulaşana kadar hiçbirinin geçemediği bir noktadır: hepsi bir döngüyü bitirmeden sonraki başlamaz. Modüller arasında karmaşık bağımlılık yoksa eşzamanlı yürütme mümkündür. Ama bu şema ölçeklenmez: kullanılabilecek düğüm sayısı genellikle az olan görev sayısını aşamaz.
- **Böl ve yönet.** Birçok sıralı algoritma özyinelemeli yazılır: çözüm, daha küçük ayrık alt problemlerin çözümlerinin birleşimidir. Örnekleri Fibonacci ve merge sort'tur. Asıl sıralamayı `mergeLists` yapar; özyineleme girdiyi giderek küçülen parçalara böler ve sıralandıkça birleştirir. **Bağımlılık grafiği çağrı grafiğiyle aynı değildir.** Sıralı yürütmede çağrılar derinlemesine sırayla yapılır. Ama aynı düzeydeki iki `mergeSort` çağrısı birbirinden bağımsızdır ve eşzamanlı yürüyebilir. 8 öğede en büyük eşzamanlılık 4 görevdir: boyu 1 olan çağrılar hemen döner ve gerçek iş değildir. Görevler çalışma sırasında **dinamik** üretilir; GPU'larda bu uzun süre olanaksızdı, Kepler'in dinamik paralelliğine kadar. Üretim, belirli bir problem boyuna ya da eşzamanlı görev sayısı düğüm sayısına ulaşana kadar sürer. Ebeveyn görev boşta beklemek yerine alt problemlerden birini kendisi çözüp $N-1$ yeni görev üretebilir. Görevlerin düğümlere eşlenmesi platforma özgüdür ve görev taşıma ile veri iletme maliyetlerine bağlıdır.
- **Geometrik ayrıştırma.** Veri dizi ya da matris gibi doğrusal yapılardaysa bir ya da birkaç boyut boyunca bölünüp görevlere atanır. Veri arasında bağımlılık yoksa problem "utanç verici derecede paraleldir". 2-B süzgeçte bloklar komşularının kenar satırlarını ister (yukarıdaki hale).
- **Özyinelemeli veri.** Ağaçlar, listeler, grafikler gibi özyinelemeli veri yapıları bu kadar dolaysız bölünemez. Yapı tek tek öğelerine ayrılır ve her öğe (ya da grup) ayrı bir düğüme atanır. Bu, eşzamanlı işlemlere izin vermek için özgün algoritmanın değiştirilmesini gerektirebilir. İşleyiş dinamik programlamayı andırır: büyük problemler küçüklerin saklanmış çözümleriyle çözülür. Örnek, bir dizinin **önek toplamlarıdır**: $S_i=\sum_{j\le i}A_j$. Sıralı algoritma doğrusaldır. Paralel sürüm giderek büyüyen grupların kısmi toplamlarını hesaplar: önce 2'li, sonra 4'lü, sonra 8'li gruplar. $\log_2N$ adımda biter, ama daha fazla iş yapar (M10.6'da Kogge–Stone ve Brent–Kung).
- **Boru hattı.** Yazılımdaki ya da donanımdaki montaj hattıdır: bir öğe, her biri ona belirli bir işlem uygulayan aşamalardan geçer. Örnekleri:
    - CPU komut boru hatları;
    - sinyal işleme (FFT → katsayı işleme → ters FFT);
    - grafik boru hattı (aydınlatma, kırpma, izdüşüm, doku kaplama);
    - Unix kabuğunda komut boruları.

    Hesabı ayrık aşamalara bölmenin yararı yalnız aşamalar eşzamanlı çalışıp farklı öğeler üzerinde işlem yaparsa ortaya çıkar. O zaman bile aşama süreleri birbirine çok yakın olmalıdır: **en yavaş aşama boru hattının hızını belirler** (Şekil 1).
- **Olay tabanlı koordinasyon.** Önceki desenlerde iletişim akışı sabittir ve tasarımda hesaba katılabilir. Bazı uygulamalarda ise görevler çalışma sırasında kararlar alarak dinamik etkileşir: iletişim desenleri ve görevlerin yürütme profili (sıklık, süre, boş zaman) önceden bilinemez. Tipik örnek **ayrık olay simülasyonudur**. Sistem bileşenleri, olay üreterek etkileşen nesnelerle modellenir. Bir olay zaman damgalı bir mesajdır: bir durum değişikliğini, bir eylem isteğini ya da bir isteğe yanıtı temsil edebilir.

<figure class="fig wide"><svg viewBox="0 0 640 210" role="img" aria-label="Eşit olmayan aşama süreli beş aşamalı bir boru hattında dört öğenin Gantt şeması"><text x="62" y="47" text-anchor="end" class="sm">aşama 1 (2)</text><text x="62" y="73" text-anchor="end" class="sm">aşama 2 (3)</text><text x="62" y="99" text-anchor="end" class="sm">aşama 3 (5)</text><text x="62" y="125" text-anchor="end" class="sm">aşama 4 (2)</text><text x="62" y="151" text-anchor="end" class="sm">aşama 5 (3)</text><path d="M70,24 L70,160" class="mute"/><text x="70" y="174" text-anchor="middle" class="sm">0</text><path d="M155,24 L155,160" class="mute"/><text x="155" y="174" text-anchor="middle" class="sm">5</text><path d="M240,24 L240,160" class="mute"/><text x="240" y="174" text-anchor="middle" class="sm">10</text><path d="M325,24 L325,160" class="mute"/><text x="325" y="174" text-anchor="middle" class="sm">15</text><path d="M410,24 L410,160" class="mute"/><text x="410" y="174" text-anchor="middle" class="sm">20</text><path d="M495,24 L495,160" class="mute"/><text x="495" y="174" text-anchor="middle" class="sm">25</text><path d="M580,24 L580,160" class="mute"/><text x="580" y="174" text-anchor="middle" class="sm">30</text><rect x="70.0" y="33" width="34.0" height="20" class="soft-a" stroke="currentColor" stroke-width="2.2"/><text x="87.0" y="47" text-anchor="middle" class="sm">0</text><rect x="104.0" y="59" width="51.0" height="20" class="soft-a" stroke="currentColor" stroke-width="2.2"/><text x="129.5" y="73" text-anchor="middle" class="sm">0</text><rect x="155.0" y="85" width="85.0" height="20" class="soft-a" stroke="currentColor" stroke-width="2.2"/><text x="197.5" y="99" text-anchor="middle" class="sm">0</text><rect x="240.0" y="111" width="34.0" height="20" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="257.0" y="125" text-anchor="middle" class="sm">0</text><rect x="274.0" y="137" width="51.0" height="20" class="soft-a" stroke="currentColor" stroke-width="0.6"/><text x="299.5" y="151" text-anchor="middle" class="sm">0</text><rect x="104.0" y="33" width="34.0" height="20" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="121.0" y="47" text-anchor="middle" class="sm">1</text><rect x="155.0" y="59" width="51.0" height="20" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="180.5" y="73" text-anchor="middle" class="sm">1</text><rect x="240.0" y="85" width="85.0" height="20" class="soft-b" stroke="currentColor" stroke-width="2.2"/><text x="282.5" y="99" text-anchor="middle" class="sm">1</text><rect x="325.0" y="111" width="34.0" height="20" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="342.0" y="125" text-anchor="middle" class="sm">1</text><rect x="359.0" y="137" width="51.0" height="20" class="soft-b" stroke="currentColor" stroke-width="0.6"/><text x="384.5" y="151" text-anchor="middle" class="sm">1</text><rect x="138.0" y="33" width="34.0" height="20" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="155.0" y="47" text-anchor="middle" class="sm">2</text><rect x="206.0" y="59" width="51.0" height="20" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="231.5" y="73" text-anchor="middle" class="sm">2</text><rect x="325.0" y="85" width="85.0" height="20" class="soft-p" stroke="currentColor" stroke-width="2.2"/><text x="367.5" y="99" text-anchor="middle" class="sm">2</text><rect x="410.0" y="111" width="34.0" height="20" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="427.0" y="125" text-anchor="middle" class="sm">2</text><rect x="444.0" y="137" width="51.0" height="20" class="soft-p" stroke="currentColor" stroke-width="0.6"/><text x="469.5" y="151" text-anchor="middle" class="sm">2</text><rect x="172.0" y="33" width="34.0" height="20" class="ca-f" stroke="currentColor" stroke-width="0.6"/><text x="189.0" y="47" text-anchor="middle" class="sm" style="fill:#fff">3</text><rect x="257.0" y="59" width="51.0" height="20" class="ca-f" stroke="currentColor" stroke-width="0.6"/><text x="282.5" y="73" text-anchor="middle" class="sm" style="fill:#fff">3</text><rect x="410.0" y="85" width="85.0" height="20" class="ca-f" stroke="currentColor" stroke-width="2.2"/><text x="452.5" y="99" text-anchor="middle" class="sm" style="fill:#fff">3</text><rect x="495.0" y="111" width="34.0" height="20" class="ca-f" stroke="currentColor" stroke-width="2.2"/><text x="512.0" y="125" text-anchor="middle" class="sm" style="fill:#fff">3</text><rect x="529.0" y="137" width="51.0" height="20" class="ca-f" stroke="currentColor" stroke-width="2.2"/><text x="554.5" y="151" text-anchor="middle" class="sm" style="fill:#fff">3</text><text x="325.0" y="190" text-anchor="middle" class="sm">zaman</text><text x="70" y="18" class="sm">4 öğe, beş aşama; kalın çerçeve: toplam süreyi belirleyen zincir. Toplam 30 = Σd + (n − 1)·max d = 15 + 3·5</text></svg><figcaption>Şekil 1 — Boru hattında en yavaş aşama (5) hızı belirler: ilk öğe 15 birimde çıkar, sonrakiler her 5 birimde bir. 100 öğe için 15 + 99·5 = 510 (hesaplandı).</figcaption></figure>

:::türetim Boru hattının süresi
$s$ aşamalı bir boru hattında aşama süreleri $d_1,\dots,d_s$ olsun ve $n$ öğe işlensin. Öğe $i$'nin aşama $k$'yı bitirme anı $C_{i,k}=\max(C_{i-1,k},C_{i,k-1})+d_k$'dır: aşama önceki öğeyi bitirmeli, öğe de önceki aşamayı bitirmiş olmalıdır. En yavaş aşama $d_{\max}$ ise ilk öğe $\sum d_k$'da çıkar, sonrakiler en yavaş aşamanın hızında akar:
$$T(n)=\sum_kd_k+(n-1)\,d_{\max}.$$
Bu, en yavaş aşamaya kadar akışın tıkanmadığı (tamponların yettiği) durum içindir. Süreler (2, 3, 5, 2, 3) ve 100 öğe ile $15+99\cdot5=510$; dengeli bir boru hattı (beşi de 3) $15+99\cdot3=312$ verirdi. Verim $1/d_{\max}$ öğe/birimdir. Darboğaz aşamasını **çoğaltmak** (iki kopya, öğeler dönüşümlü) etkin süresini yarıya indirir.
:::

### Program yapısı desenleri {#yapi}

Desenler yalnız ayrıştırmayı seçmeye değil, programı geliştirmeye de yardım eder. Program yapısı desenleri iki gruba ayrılır:

- **Küresel paralel, yerel sıralı (GPLS):** uygulama birçok görevi eşzamanlı yürütür, her görev kendi içinde sıralıdır. SPMD, MPMD, efendi–işçi ve map-reduce bu gruptadır.
- **Küresel sıralı, yerel paralel (GSLP):** uygulama sıralı bir program gibi yürür, bazı kısımları istendiğinde paralel çalışır. Fork/join ve döngü paralelliği bu gruptadır.

**SPMD** (tek program, çok veri). Bütün düğümler aynı programı çalıştırır, ama aynı işlemleri farklı verilere uygular ve/veya program içinde farklı yürütme yolları izler. Bütün mantığın tek programda olması geliştirmeyi kolaylaştırır ve hataları azaltır; bu yüzden yaygındır. Tipik yapı şu adımlardan oluşur:

1. **Başlatma:** program platforma yayılır, thread ya da süreçlerin haberleşmesini sağlayan çalışma zamanı sistemi başlatılır.
2. **Benzersiz bir kimlik edinme:** kimlikler genellikle 0'dan numaralanır. CUDA'da olduğu gibi skaler değil vektör de olabilir (`blockIdx`, `threadIdx`). Kimliğin ömrü thread'in ömrünü izler; kalıcı ya da gerektiğinde dinamik üretilmiş olabilir.
3. **Programı çalıştırma:** kimliğe karşılık gelen yürütme yolu izlenir; iş ya da veri dağıtımı ve rol ayrımı buna dahildir.
4. **Kapatma:** thread ya da süreçler kapatılır, kısmi sonuçlar nihai yanıtta birleştirilir.

**MPMD** (çok program, çok veri). SPMD çoğu senaryoyu kapsar, ama iki durumda yetersiz kalır. Biri platformun heterojen olmasıdır: düğümün mimarisine göre farklı yürütülebilir dosyalar gerekir. Öteki bellek gereksinimlerinin, her düğüme yalnız gerekli mantığın yüklenmesini zorunlu kılacak kadar ağır olmasıdır. MPMD, belki farklı araç zincirleriyle üretilmiş farklı yürütülebilir dosyaları tek uygulamada birleştirir. Yukarıdaki adım dizisi yine izlenebilir.

**Efendi–işçi.** Düğümlerin rolleri ikiye ayrılır. Efendinin görevleri şunlar olabilir:

- işçilere iş dağıtmak;
- sonuçları toplamak;
- işçiler adına G/Ç yapmak (veri göndermek, dosyaya erişmek);
- kullanıcıyla etkileşmek.

En basit biçimi tek efendi ve çok işçidir. Bu düzen ölçeklenmez, çünkü efendi darboğaz olur. Çözüm, her biri makinelerin bir kısmını yöneten ve daha üst bir efendiye bağlı efendilerden oluşan **hiyerarşik** düzenlerdir. Hesap, düğümler arası iletişim gerektirmeyen ayrık parçalara bölünebiliyorsa efendi–işçi doğal biçimde uygulanır. Ek bir yararı **örtük yük dengelemedir**: efendi işi boşalan düğümlere verdiği için dengesizlik çok azdır. İş tanımları çok özel (bilinen bir fonksiyonun parametreleri) ya da çok genel (her türlü hesabı taşıyan sınıf örnekleri) olabilir.

**Map-reduce.** Efendi–işçinin popüler bir türevidir; Google'ın arama motoru için gerçekleştirimiyle yaygınlaştı (Dean ve Ghemawat 2004). Bağlam, büyük bir bağımsız veri koleksiyonuna bir fonksiyon uygulamak ("map") ve bütün kısmi sonuçları ikinci bir fonksiyonla "indirgemektir" ("reduce"). Genel biçimi altı adımdır:

1. efendi ve işçi thread'lerinin başlatılması;
2. efendinin görev ataması;
3. map yapan işçilerin veriyi okuması;
4. kısmi sonuçların kaydedilmesi;
5. reduce yapan işçilerin kısmi sonuçları okuması;
6. nihai sonucun kaydedilmesi.

Pratikte map ve reduce yapan işçiler farklı olmak zorunda değildir. Aradaki depolama kalıcı (dosya) ya da geçici (bellek tamponu) olabilir. Tipik efendi–işçiden temel farkı, biçimlendirmenin dağıtımı ve yük dengelemeyi üstlenen otomatik araçların kullanımına izin vermesidir.

:::uyarı Kaynak düzeltmeleri (lect_02 slayt 29 ve 52)
**Slayt 29.** Önek toplamının sıralı kodunda `S[i] += S[i-1] + A[i];` yazılmış. `+=`, başlatılmamış `S[i]`'ye ekleme yapar; doğrusu `S[i] = S[i-1] + A[i];`'dir.

**Slayt 52.** Map, scan ve reduce'un üçü de "girdinin her öğesine bir fonksiyonu sırayla uygulamak" diye tanımlanmış; tanımlar birbirinden ayırt edilemiyor. Doğru tanımlar şunlardır:

- **Map:** $y_i=f(x_i)$. Her çıktı yalnız kendi girdisine bağlıdır; öğeler **bağımsızdır**, "sırayla" değil, aynı anda işlenebilir.
- **Scan** (önek): $y_i=y_{i-1}\oplus x_i$, yani $y_i=x_0\oplus\dots\oplus x_i$. Her çıktı önceki bütün girdileri birleştirir.
- **Reduce:** yalnız son değer, $x_0\oplus\dots\oplus x_{n-1}$.

Scan ve reduce'un paralelleştirilebilmesi için $\oplus$ **birleşmeli** olmalıdır (M10.5, M10.6; M9.18'de doğrusal özyineleme).
:::

**Fork/join.** Paralel algoritma çalışma sırasında görevlerin dinamik yaratılmasını (fork) gerektirdiğinde kullanılır. Çocuk görevler (süreç ya da thread) ebeveyn devam etmeden önce genellikle bitmelidir (join). Görevler yeni thread ya da süreç açarak ya da var olan bir **thread havuzunu** kullanarak çalıştırılabilir. İkincisi thread yaratma ek yükünü en aza indirir ve thread sayısını çekirdek sayısına eşleyerek makinenin kaynaklarını en iyi biçimde yönetebilir. Modern fork/join çalışma zamanları (Cilk, TBB, Java ForkJoinPool) **iş çalma** kullanır. Her thread'in kendi görev kuyruğu vardır; boşalan bir thread başkasının kuyruğunun öbür ucundan görev çalar. Blumofe ve Leiserson (1999), iş çalmanın beklenen sürede $T_p\le W/p+O(S_\infty)$ verdiğini gösterdi; bu, yukarıdaki açgözlü sınırın dağıtık bir gerçekleştirimidir.

**Döngü paralelliği.** Yazılımı çok çekirdekli mimarilere taşımak dev bir iştir. Döngü paralelliği, yürütme süresine hâkim olan döngüleri paralelleştirerek var olan sıralı kodu taşımayı sağlar; kısacası, `for` döngülerini eniyilemek. Bu desen özellikle OpenMP için önemlidir. Orada döngüler, programcının yönergeleriyle (`#pragma omp parallel for`) yarı otomatik paralelleştirilir. Kullanımı sınırlıdır: bir probleme baştan yeni bir paralel çözüm tasarlamayı değil, sıralı bir çözümün paralele evrimini hedefler. Başarım kazançları genellikle küçüktür, ama geliştirme çabası da öyle.

**Sonuç.** Belirli bir ayrıştırma deseni için programı yapılandırmanın en iyi yolu uygulamaya çok bağlıdır. Bazı platformlar belirli bir program yapısı dayatır: MPI SPMD/MPMD'yi, OpenMP döngü paralelliğini teşvik eder. Yine de belirli bir ayrıştırma desenini gerçekleştirmeye bazı program yapıları daha uygundur. Ders bunu şöyle eşliyor (✓: uygun):

| program yapısı | görev | böl-yönet | geometrik | özyin. veri | boru hattı | olay |
|---|---|---|---|---|---|---|
| SPMD | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MPMD | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| efendi–işçi | ✓ | ✓ | ✓ | ✓ | | |
| map-reduce | | ✓ | ✓ | ✓ | | |
| fork/join | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| döngü paralelliği | | ✓ | ✓ | | | |

SPMD, MPMD ve fork/join genel amaçlıdır. Efendi–işçi ve map-reduce bağımsız iş parçaları ister; düzenli akan veriye (boru hattı) ya da dinamik etkileşime (olay) uymazlar. Döngü paralelliği yalnız döngülerle ifade edilen veri bölünmelerine uyar.

### Yük dengeleme {#yuk}

Eşleme adımının ilk amacı yük dengesidir. Görevlerin maliyetleri önceden biliniyorsa ve eşitse, işi eşit bloklara bölmek (blok dağıtım) yeter. Mandelbrot kümesi bunun tersinin klasik örneğidir. Her piksel kaçış iterasyon sayısı kadar iş ister ve kümenin içindeki pikseller en çok iterasyonu harcar. Satırların maliyetleri bu yüzden çok farklıdır: aşağıdaki deneyde en pahalı satır ortalamanın 2.4 katı, en ucuzu 0.06 katı. Dağıtım seçenekleri şunlardır:

- **Blok:** thread $k$ ardışık $\lceil H/p\rceil$ satırı alır. Pahalı satırlar ortada toplandığı için ortadaki thread'ler darboğaz olur: 8 thread'de hızlanma 3.7.
- **Döngüsel:** satır $i$ → thread $i\bmod p$. Her thread kümenin her bölgesinden örnek alır ve yükler dengelenir: 7.7. Bedeli komşu satırların farklı thread'lerde olmasıdır (önbellek yerelliği azalır).
- **Döngüsel blok:** $c$ satırlık parçalar döngüsel dağıtılır; yerellik ile denge arasında bir ödünleşimdir.
- **Dinamik:** boşalan thread sıradaki $c$ satırlık parçayı alır (paylaşılan bir atomik sayaçla; M10.2). Maliyetler önceden bilinmese de dengeler, ama her alışın bir ek yükü vardır. Parça sayısı thread sayısına göre az olursa son parçalar dengesizlik yaratır. 32 thread ve $c=4$ ile yalnız 24 parça olduğu için hızlanma 10'a düşüyor. Parçalı dağıtımların bir tavanı daha vardır. En pahalı parça bölünmez ve tek bir thread'e düşer; hızlanma, thread sayısı ne olursa olsun, toplam iş / en pahalı parçanın maliyeti ile sınırlıdır. Burada $c=4$ için bu oran 10.2'dir ve döngüsel blok ile dinamik eğrilerin 12 thread civarında doyması bundandır. İş–yol modelinin diliyle, en pahalı parça bu dağıtımın yoludur.

::lab l3

### Özet {#ozet}

- Frekans güç duvarına takıldı ($P\propto f^3$). Transistörler çekirdeklere ve GPU'lara gidiyor; GPU'lar verim ve watt başına işlemde üstün, ama PCIe aktarımı bir veri eşiği yaratıyor.
- Flynn sınıflaması SISD/SIMD/MISD/MIMD'dir. MIMD paylaşımlı bellekli (NUMA, SMP, efendi–işçi) ya da dağıtık bellekli olabilir. Modern işlemciler hem MIMD hem SIMD'dir; ders Cell, Kepler, APU/HSA, TILE-Gx ve Xeon Phi'yi örnekliyor.
- Hızlanma ve verim ölçülürken altın kurallar uygulanır: aynı platform ve en hızlı sıralı kod.
- Amdahl: $1/((1-\alpha)+\alpha/N)\le1/(1-\alpha)$; karıncalar ve filler. Gustafson: $(1-\alpha)+N\alpha$. İkisi güçlü ve zayıf ölçeklemedir. Karp–Flatt deneysel seri kesri verir.
- İş–yol modeli: $\max(W/p,S_\infty)\le T_p\le W/p+S_\infty$. Paralellik $W/S_\infty$'dur; merge sort'ta $\sim\log n$, ağaç indirgemede $\sim n/\log n$.
- PCAM adımları bölümleme, iletişim, birleştirme ve eşlemedir. Altı ayrıştırma deseni vardır (görev, böl ve yönet, geometrik, özyinelemeli veri, boru hattı, olay tabanlı). Program yapıları GPLS (SPMD, MPMD, efendi–işçi, map-reduce) ve GSLP'dir (fork/join, döngü paralelliği).
- Boru hattının süresi $\sum d+(n-1)d_{\max}$'dır. Yük dengeleme için blok, döngüsel ve dinamik dağıtım kullanılır.
