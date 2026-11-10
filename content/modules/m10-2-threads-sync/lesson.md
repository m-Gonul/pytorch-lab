### Süreçten thread'e {#surec}

Birden çok programın paralel yürütülmesi 1960'lardan beri olağandır. MIT'nin Compatible Time Sharing System (CTSS) işletim sistemi bunu yaygınlaştırdı. İşletim sistemi o anda çalışan programı keser ve CPU'nun denetimini başka bir programa verir. CPU zamanını "paylaştıran" bu geçişi üç şey tetikleyebilir:

- bir saatin ürettiği düzenli donanım kesmeleri;
- düzensiz donanım kesmeleri, örneğin ilgi isteyen bir aygıttan;
- işletim sistemine yapılan bir çağrı, örneğin G/Ç isteği.

Çalışan her program bir **süreçtir** (process): işletim sisteminin programı yönetmek için gereksindiği kodu, veriyi, kaynakları ve yürütme durumunu sarmalayan bir varlık. Her süreç CPU'yu bir "zaman dilimi" boyunca kullanır, sonra işletim sistemine bırakır.

Zaman paylaşımı kaynakları verimli kullanır, ama tek bir sürecin işini hızlandıramaz. Tersine, bir sürecin yavaşlamasına yol açabilir: ona verilen CPU zamanını doğrudan azaltır, görev değiştiricinin (dispatcher) sık çalışmasıyla da kullanılabilir zamanı dolaylı olarak seyreltir. Bir programın daha büyük bir hesap payı alması için (öncelik ayarları dışında) birden çok sürece bölünmesi gerekir. Çok çekirdekli bir platformda da aynısı geçerlidir.

**fork().** Birden çok süreç yaratmanın özgün yolu `fork()` sistem çağrısıdır. Çağrı, sürecin belleğinin o anki birebir kopyasını yaratır. Ebeveyn ve çocuk, çağrıdan sonraki satırdan devam eder; yalnız `fork()`'un dönüş değeri farklıdır: çocukta 0, ebeveynde çocuğun süreç kimliği (pid). Bu fark, iki sürecin farklı işler yapmasını sağlayan mantığın temelidir.

İki soru akla gelir. Kod neden kopyalanıyor, kod değişmez değil mi? Kendi kodunu çalışırken değiştiren programlar vardır: virüsler en bilinen örnektir, ama işletim sistemi çekirdekleri ve JIT derleyicileri de bunu yapar. Çocuk ebeveynin bütün verisinin özel kopyasına sahipse ikisi nasıl haberleşir? Borular (pipe) gibi mekanizmalar vardır, ama hiçbiri ortak değişkenlere erişmek kadar kolay değildir. Bir sürecin görüntüsünü ayırmanın ve kopyalamanın ek yükü `fork()`'u pahalı bir eşzamanlılık yolu yapar. Daha hafif bir şey gerekir: **thread**.

:::uyarı Kaynak düzeltmesi (slayt 7 ve 9): fork'un dönüş değeri ve kopyalama
Slayt 7 çocuk sürecin `fork()`'un dönüş değerini "almadığını" söylüyor. Çocuk da bir dönüş değeri alır: **0**. Ebeveyn ise çocuğun pid'ini alır, hata durumunda −1. Slayttaki kodun `if (fork() == 0)` ile çocuğu ayırt edebilmesi bu yüzdendir.

Ayrıca modern işletim sistemleri `fork()`'ta belleği hemen kopyalamaz. **Yazınca kopyala** (copy-on-write) kullanırlar: sayfalar iki süreç arasında salt okunur paylaşılır ve ancak biri yazınca o sayfa kopyalanır. `fork()` yine de sayfa tablolarını çoğalttığı için thread yaratmaktan pahalıdır, ama bütün belleğin kopyalandığı izlenimi doğru değildir.
:::

### Thread nedir? {#thread}

Bir thread hafif bir süreç olarak düşünülebilir. Daha kesin bir tanımla, işletim sistemi zamanlayıcısının ayrı bir birim olarak yönettiği bir yürütme yoludur, bir komut dizisidir. Bir süreçte birden çok thread olabilir. Thread'ler `fork()` mekanizmasının ek yükünü, yalnız zorunlu olanı kopyalayarak azaltır: **çalışma zamanı yığınını**. Yığın iki thread arasında paylaşılamaz, çünkü çağrılan fonksiyonların etkinleştirme kayıtlarını (activation record) içerir. Paylaşılan bir yığında, örneğin bir fonksiyondan dönüş, onu çağırandan farklı bir noktaya gidebilirdi. Kod, küresel veri ve yığın bellek (heap) paylaşılır. Ayrı süreçlerin bellek koruması ve esnek zamanlama gibi yararları vardır, ama çoğu durumda dezavantajları ağır basar.

**Thread'ler ne işe yarar?** Genellikle grafik arayüz programlamayla anılırlar: kullanıcının başlattığı bir işi yaparken arayüzün yanıt vermeye devam etmesi, arayüze ayrılmış bir thread olmadan mümkün değildir. Ama kullanım alanları çok daha geniştir:

1. **Başarımı artırmak:** iş birden çok thread'e bölünerek birden çok çekirdek kullanılır. Her çekirdeğin yalnız ek hesap gücü değil, kendi alt düzey önbellekleri de vardır. Ancak ayrı önbellekler bir sorun da yaratır: aynı verinin birden çok kopyası.
2. **Arka plan görevleri:** etkileşimli uygulamalar, kullanıcıya yanıt vermeye devam ederken etkileşim gerektirmeyen işleri yapar.
3. **Eşzamansız işlem:** bir sunucuya ağ üzerinden istek göndermek büyük gecikme içerir. İşlemi ayrı bir thread'e verip yanıtı beklemeden yararlı iş yapmaya devam etmek kaynak kullanımını iyileştirir.
4. **Program yapısını iyileştirmek:** bir oyun ekranı yeniden çizme, ses çalma, kullanıcı girdisi ve strateji gibi farklı sıklıklarda çalışan periyodik görevler içerir. Hepsini tek bir döngüye koyup ekran yenileme hızında çalıştırmak programı karmaşıklaştırır. Ayrı thread'ler daha iyi yapılandırılmış ve bakımı kolay kod verir.

**C++ ve thread'ler.** Thread yaratma ve yönetme uzun süre standart C++ kütüphanesinin parçası değildi. Boşluğu üçüncü taraf kütüphaneler doldurdu:

- **pthreads** (POSIX thread'leri): thread desteğinin ortak paydası sayılan C kütüphanesi.
- **winThreads:** yalnız Windows için.
- **Qt thread'leri:** kullanımı kolay, zengin bir API'si olan platformlar arası Qt kütüphanesinin parçası.

C++11 standardı kütüphaneye bir `thread` sınıfı ekledi. POSIX sistemlerinde std::thread genellikle pthreads üzerine kuruludur.

POSIX thread'leri bir dilden bağımsız bir yürütme modelidir. Bir programın zamanda örtüşen birden çok iş akışını denetlemesini sağlar ve IEEE POSIX.1c (1995) ile tanımlanmıştır. FreeBSD, Linux, macOS, Android ve Solaris gibi sistemlerde genellikle `libpthread` olarak gelir; Windows için de uygulamaları vardır. Yaklaşık 100 fonksiyonun hepsi `pthread_` önekiyle başlar ve beş grupta toplanır:

- thread yönetimi,
- mutex'ler,
- koşul değişkenleri,
- okuma–yazma kilitleri ve bariyerler,
- dönen kilitler (spinlock).

POSIX semaforları pthreads standardının değil gerçek zaman eklerinin (POSIX.1b) parçasıdır ve `sem_` önekini taşır.

### C++11 thread'leri {#cpp11}

```cpp
void hello() { cout << "Hello from thread " << this_thread::get_id() << "\n"; this_thread::sleep_for(1s); }
int main() { thread t(hello); cout << "Hello from the main thread\n"; t.join(); }
```

- **Yaratma ve çalıştırma:** `thread t(hello)` tek çağrıyla thread'i yaratır ve `hello`'yu (çalıştırıcı, işçi) yürütmeye başlar. Fonksiyon değer döndürmez.
- **Bekleme:** ebeveynin çocuklarını beklemesi iyi bir alışkanlıktır; kaynakların bırakılmasını ve dosyaların kapanmasını güvenceye alır. Bu, **bloklayan** `join()` çağrısıyla yapılır. Bloklayan bir çağrı, yaptığı iş tamamlanana (ya da yapılamayacağı kesinleşene) kadar dönmez. Bloklamayan bir çağrı ise işin yapılması için "emri verir vermez" döner. Bloklayan çağrıların genellikle bir zaman aşımı parametresi vardır.
- **this_thread:** `this_thread` ad alanı, çalışan thread'i sorgulamayı ya da etkilemeyi sağlar. `get_id()` kimliği verir; `sleep_for(1s)` bir saniye uyutur (`s` soneki `chrono`'dan gelir).
- **Kopyalanamazlık:** thread nesneleri kopyalanamaz; bir fonksiyon thread kurup çağırana döndürecekse bunu işaretçi olarak ya da nesneyi **taşıyarak** (`std::move`) yapar.
- **Ayırma:** `detach()` thread'i ebeveyninden ayırır ("ateşle ve unut"). Thread, onu başlatan nesne yok edilse bile çalışmayı sürdürür. Ancak `main` biterse süreç sona erer ve bütün thread'ler onunla biter.

**Veri geçirmek.** Örnekteki thread'e veri verilmedi; bu tipik değildir. Veri paralel uygulamalarda aynı fonksiyonu farklı veriler üzerinde çalıştıran bir thread koleksiyonu isteriz. İki seçenek vardır:

- **Küresel değişkenler.** Tasarım açısından kötü bir fikirdir. Üstelik `get_id()` bir sayı değil bir nesne döndürür: yalnız iki thread'in aynı olup olmadığını karşılaştırmaya yarar, veriyi bölmek için kullanılamaz (kimlikler 0, 1, 2, … gibi ardışık tamsayılar değildir).
- **Thread fonksiyonuna parametre geçirmek.** Önerilen yoldur: `thread` yapıcısı istediği kadar parametre alır ve onları fonksiyona iletir. Parametreler **değerle** kopyalanır; başvuru geçirmek için `std::ref`/`std::cref` gerekir.

Ders kodlarındaki fonksiyon nesneleri (functor) yapıcıya bu yüzden `ref()` ile verilir. Değerle verilseler hesap doğru parametrelerle yapılırdı, ama ana thread'in yarattığı nesneler değil kopyaları kullanılırdı ve sonuçlara ulaşılamazdı.

**ThreadGuard.** Ders kodlarındaki `ThreadGuard` sınıfı, kapsamdan çıkarken thread'i otomatik `join` eden bir RAII sarmalayıcısıdır. İki ayrıntısı öğreticidir:

- Yapıcısı `explicit`'tir. Tek argümanla çağrılabilen bir yapıcı örtük bir **dönüştürme yapıcısı** olur; `explicit` beklenmedik örtük dönüşümleri derleme hatasına çevirir.
- Kopya yapıcısı ve kopya atama işleci `= delete` ile silinmiştir. Aksi hâlde kopyalar, sarmaladıkları thread yok edildikten sonra kapsamları dışında yaşayabilirdi.

C++20'nin `std::jthread` sınıfı bu davranışı standart hâle getirir: yıkıcısında otomatik `join` eder.

:::uyarı Kaynak düzeltmesi (slayt 31): join'den sonra thread nesnesi yok edilmez
Slayt "join() çağrıldığında karşılık gelen nesnenin yok edildiğini" söylüyor. `join()` nesneyi yok etmez; nesne kapsamının sonuna kadar yaşar ama artık **birleştirilebilir değildir** (`joinable()` false). Aynı thread için ikinci bir `join()` ya da `detach()` istisna fırlatır. Asıl tehlike tersidir: birleştirilebilir bir `std::thread` nesnesi (ne `join` ne `detach` edilmiş) yok edilirse program `std::terminate` ile sonlanır. ThreadGuard ve `jthread` bunun için vardır.
:::

### Thread'ler arasında veri paylaşmak {#paylasim}

Veri paylaşımı, thread'lerin aynı bellek adreslerine ya da program değişkenlerine erişmesidir. İki senaryo vardır:

- Thread'ler veriyi **yalnız okuyorsa** çok thread'li kod yazmak görece kolaydır.
- Thread'lerin **okuyup yazdığı** paylaşılan değişkenler varsa iş çok zorlaşır.

Sorular şunlardır. Bir thread, başka bir thread'in o anda yazdığı bir değişkeni okursa ne olur? Thread'ler arasında belirli bir okuma/yazma sırası varsayabilir miyiz? (Hayır, asla.) Bir thread paylaşılan bir değişkeni okuyup işlerken başka bir thread onu güncellerse ne olur?

Bir thread paylaşılan bir değişkene yazarken bunu yalıtılmış olarak yapmalıdır: başka hiçbir thread aynı anda yazmamalı, tutarsız bir durum okunabileceği için okumalar da engellenmelidir. Yazma işlemlerini "atomik" (bölünmez) yapmak çoğu zaman yetmez. Bir nesne ya da veri yapısı, durumu değiştiren bir işlem süresince tamamen **kilitlenmeli**, ya da erişim sıkı biçimde sıraya sokulmalıdır.

**Banka hesabı.** İki thread aynı hesap nesnesi üzerinde çalışsın: biri maaş yatırıyor, öteki bir faturayı ödüyor ya da başka bir yatırma yapıyor. Her işlem bakiyeyi **okur**, yeni değeri **hesaplar**, **yazar**. İki thread aynı eski bakiyeyi okursa, sonra yazanın değeri öncekinin değişikliğini siler (Şekil 1). Bu, olası sonuçlardan yalnız biridir; hesabın doğru güncellendiği sıralamalar da vardır (biri tamamen bittikten sonra öteki başlarsa). Sorun, sonucun **önceden kestirilemez** olmasıdır.

<figure class="fig wide"><svg viewBox="0 0 640 230" role="img" aria-label="İki thread'in aynı hesaba eşzamanlı para yatırması: oku, topla, yaz adımlarının bir iç içe geçmesi ve kayıp güncelleme"><text x="20" y="48" class="sm">T1 (yatır 1000)</text><text x="20" y="98" class="sm">T2 (yatır 500)</text><text x="20" y="148" class="sm">bakiye</text><rect x="150" y="30" width="69" height="28" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="184.5" y="48" text-anchor="middle" class="sm">oku: 10000</text><text x="184.5" y="148" text-anchor="middle" class="sm">10000</text><rect x="225" y="80" width="69" height="28" class="soft-b" stroke="currentColor" stroke-width="0.8"/><text x="259.5" y="98" text-anchor="middle" class="sm">oku: 10000</text><text x="259.5" y="148" text-anchor="middle" class="sm">10000</text><rect x="300" y="30" width="69" height="28" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="334.5" y="48" text-anchor="middle" class="sm">topla: 11000</text><text x="334.5" y="148" text-anchor="middle" class="sm">10000</text><rect x="375" y="80" width="69" height="28" class="soft-b" stroke="currentColor" stroke-width="0.8"/><text x="409.5" y="98" text-anchor="middle" class="sm">topla: 10500</text><text x="409.5" y="148" text-anchor="middle" class="sm">10000</text><rect x="450" y="30" width="69" height="28" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="484.5" y="48" text-anchor="middle" class="sm">yaz: 11000</text><text x="484.5" y="148" text-anchor="middle" class="sm">11000</text><rect x="525" y="80" width="69" height="28" class="soft-b" stroke="currentColor" stroke-width="0.8"/><text x="559.5" y="98" text-anchor="middle" class="sm">yaz: 10500</text><text x="559.5" y="148" text-anchor="middle" class="sm">10500</text><path d="M150,160 L594,160" class="mute"/><text x="375" y="178" text-anchor="middle" class="sm">zaman →</text><text x="20" y="200" class="sm">Bu sırada T2, T1 ile aynı eski bakiyeyi okudu ve sonra yazdı: son bakiye 10500 (doğrusu 11500).</text><text x="20" y="218" class="sm">20 sıranın sonuçları — 10500: 9, 11000: 9, 11500: 2.</text></svg><figcaption>Şekil 1 — Yarış koşulu: iki yatırma da aynı eski bakiyeyi okudu, sonra yazan kazandı. Üç adımlı modelde 20 sıranın yalnız 2'si doğru (hesaplandı).</figcaption></figure>

İki kavram ayrılmalıdır:

- **Veri yarışı** (data race): iki ya da daha çok thread aynı bellek konumuna **eşzamanlı** erişir ve erişimlerden en az biri yazmadır. C ve C++'ta veri yarışı **tanımsız davranıştır**: derleyici böyle bir şeyin olmadığını varsayabilir. Banka örneğinde 2. ve 3. adım aynı anda olsaydı yarıda kalmış bir okuma (torn read) bile görülebilirdi.
- **Yarış koşulu** (race condition): sonucun olayların göreli zamanlamasına bağlı olmasından doğan hatalı program davranışıdır. Veri yarışı olmadan da yarış koşulu olabilir, tersi de doğrudur.

Bazı veri yarışları yarış koşuluna yol açar; şans eseri doğru çıktı veren bir veri yarışı da olabilir. Bütün veri yarışlarını gidermek yarış koşullarını gidermeye yetmez. Ders bunu, bakiyeyi okumayı ve yazmayı ayrı ayrı atomik yapan bir `deposit` sürümüyle gösteriyor: `temp` yereldir ve bakiyeye her erişim atomiktir, yani veri yarışı yoktur. Ama okuma ile yazma arasında başka bir thread araya girebilir, dolayısıyla yarış koşulu aynen sürer. Okuma–hesaplama–yazmanın **bütünü** bölünmez yapılınca ikisi de ortadan kalkar; kilitler ve mutex'ler bunu sağlar.

:::türetim Kaç iç içe geçme var, kaçı doğru?
İki thread'in her biri $k$ adımlık bir dizi yürütsün. Her thread'in kendi adımlarının sırası korunarak oluşabilecek bütün birleşik sıralar (iç içe geçmeler), $2k$ konumdan hangi $k$'sının birinci thread'e ait olduğunu seçmekle belirlenir:
$$\binom{2k}{k}.$$

**İki adımlı model** (LOAD, STORE(+1)): $\binom42=6$ iç içe geçme. Doğru sonuç (2) ancak bir thread'in STORE'u ötekinin LOAD'undan önce olursa çıkar; bu yalnız iki sırada olur, AABB ve BBAA. Kalan 4 sıra güncellemenin birini kaybeder.

**Üç adımlı model** (oku, artır, yaz): $\binom63=20$ iç içe geçme ve yine yalnız 2'si doğrudur. Banka örneğinde üç farklı son bakiye vardır: 11 500 (2 sıra), 11 000 (9 sıra), 10 500 (9 sıra).

$n$ artırımlık döngülerde sıra sayısı $\binom{2kn}{kn}$ ile patlar; doğru olanların payı hızla sıfıra gider. Test ederek bir yarışı "bulmamak" bu yüzden hiçbir şey kanıtlamaz.
:::

Aşağıdaki deney bunu ölçüyor. 2 thread'in her biri 50 kez artırma yapıyor ve önalım olasılığı adım başına 0.05. Bu koşullarda 1000 çalıştırmanın yalnız %4.4'ü doğru sonucu (100) veriyor; ortalama 30 güncelleme kayboluyor. "Atomik oku + atomik yaz" sürümü veri yarışını gideriyor ama aynı kayıp dağılımını veriyor: yarış koşulu sürüyor. `fetch_add` her sıralamada 100 veriyor.

::lab l1

:::uyarı x86'da hizalı 8 baytlık okuma ve yazma zaten atomik değil mi?
Modern x86 hizalı 8 baytlık yüklemeleri ve saklamaları gerçekten atomik yapar. Ama bu varsayımla kod yazılmamalıdır. Aynı adrese bir thread'in yazıp ötekinin okuması C/C++'ta tanımsız davranıştır ve derleyici bunun olmadığını varsayabilir. Donanım güvencesi şu nedenlerle pek bir şey ifade etmez:

- 8 bayttan büyük veri türleri vardır.
- Derleyici bellek işlemlerini yeniden sıralar, bir değişkeni yazmaçta tutar, okumaları birleştirir ya da bölebilir.

Az sayıda paylaşılan kaynakta (sayaçlar, toplayıcılar) **atomik türler** kullanılmalıdır. Atomik işlemleri donanım komutları destekler (x86'da `lock xadd`, `cmpxchg`); yettikleri yerde hızlı ve verimlidirler.

`std::atomic` derleyiciye bir bildirimdir. Programcı neyin atomik olması gerektiğini bilir, ama derleyici her thread'in kodunu ayrı derler. `atomic`, o değişken üzerindeki işlemlerin bölünmemesini ve yeniden sıralamanın sınırlanmasını sağlar.

GPU'da bir değişkeni çok sayıda thread'in aynı anda okuması, nerede saklandığına bağlı olarak ciddi bir darboğaz olabilir. Yayın okumasına eniyilenmiş özel bir bellek türü (sabit bellek) bu durum içindir (M10.4).
:::

### Kilit, mutex, semafor {#kilit}

Çözüm, "yarışmayı" ortadan kaldırmaktır: her thread paylaşılan kaynağı **dışlayıcılık** koşulunda değiştirmelidir. Değişikliği yapan kod parçasına **kritik bölge** (critical section) denir; dışlayıcılık fikri bir tür bariyer ya da kilit gerektirir. Terimler şöyledir:

- **Kilit:** koruduğu kritik bölgeye aynı anda yalnız bir thread'in girmesine izin verir. Onu alan thread bırakmalıdır. Bazı kilitler aynı thread tarafından birden çok kez alınabilir (özyinelemeli), ama aynı sayıda bırakılmalıdır.
- **Mutex** (MUTual EXclusion): kilitle aynı işi görür. Bazı sistemlerde (Windows'ta) süreçler arası, sistem çapında da olabilir. Aksi belirtilmedikçe "kilit" ile mutex kastedilir.
- **Semafor:** aynı anda $N$ thread'in girmesine izin verir. CPU, G/Ç ya da bellek gibi kritik kaynakları aynı anda kullanan thread sayısını sınırlamak için kullanılır.

C++'ın `std::mutex`'i yalnız bir sürecin thread'leri arasındadır. Süreçler arası mutex, paylaşılan bellekte POSIX `PTHREAD_PROCESS_SHARED` gibi işletim sistemi araçlarıyla kurulur.

### Tutarlılık modelleri {#tutarlilik}

Yukarıdaki sorun şöyle özetlenebilir: çok thread'li bir program sıralı bir program gibi nasıl davrandırılır? Burada **tutarlılık modelleri** devreye girer. Bir tutarlılık modeli, programcı ile sistem arasında bir sözleşmedir: programcı bellek işlemleri için kurallara uyarsa sistem belleğin tutarlı ve okuma–yazma sonuçlarının öngörülebilir olacağını güvence eder. Farklı modeller verimlilik ile katılık arasında farklı ödünleşimler sunar. Tutarlılık modelleri dağıtık paylaşımlı bellekte ve dağıtık veri depolarında (dosya sistemleri, veri tabanları, web önbellekleri) da kullanılır.

Tutarlılık, **eşevrelilikten** (coherence) farklıdır:

- **Eşevrelilik**, tek bir konuma yapılan yazmaların bütün işlemcilerce aynı sırada görülmesidir.
- **Tutarlılık**, birden çok konuma yapılan işlemlerin bütün işlemcilere göre sıralanmasıdır.

C++ ve Java gibi yüksek düzeyli diller sözleşmeyi üç yolla korur: bellek işlemlerini anlamı koruyacak biçimde alt düzey işlemlere çevirir, bazı bellek komutlarını yeniden sıralar ve gereken eşzamanlamayı `pthread_mutex_lock()` gibi kütüphane çağrılarına sarar.

**Sıralı tutarlılık** (sequential consistency, Lamport 1979) iki koşul koyar:

- Paylaşılan bellek nesneleri üzerindeki bütün olaylar **birer birer** oluyormuş gibi görünmelidir.
- Her thread açısından olaylar **program sırasında** etkili olmalıdır.

Farklı thread'lerdeki olaylar, nesnenin genel geçmişi sıralı belirtimine (tek thread'de nasıl davrandığına) uyacak biçimde kaydırılabilir. Ders iki thread'in bir FIFO kuyruğu üzerindeki zaman çizelgelerini bu biçimde sıralı bir kuyruğa denk olacak şekilde yeniden sıralıyor. Sıralı tutarlılığın bir sakıncası **bileşimsel olmamasıdır**: sıralı tutarlı iki modülü birleştirmek sıralı tutarlı bir bütün vermeyebilir. Ders aynı sırayı ikinci bir kuyrukla çoğaltınca iki zaman çizelgesinin birlikte sıralı tutarlı olamayacağını gösteriyor.

**Doğrusallaştırılabilirlik** (linearizability, Herlihy ve Wing 1990) daha katıdır. Olaylar birer birer olmanın yanında **anlık** gerçekleşiyormuş gibi görünmelidir. Bu, **doğrusallaştırma noktalarıyla** modellenir: bir yöntem çağrısının süresi içinde etkisinin gerçekleştiği kabul edilen bir an. İki sonucu vardır:

- Yöntemler doğrusallaştırma noktalarına göre tam sıralanabilir. Nokta, çağrı ile dönüş arasında keyfi seçilebilir; ortaya çıkan sıra sıralı bir yürütmeye denk olmalıdır.
- Kilitleme zorunlu değildir (başarım için kaçınılmalıdır), ama bir nesneyi yöntem süresince kilitlemek etkinin herkes için anlık olması demektir. En basit yol budur; uygulamada kilitlerin en yaygın araç olmasının nedeni de budur.

:::uyarı Kaynak düzeltmeleri (slayt 67, 117 ve 132): donanım sıralı tutarlı değildir
Slayt 67 modern çok çekirdekli CPU'ların "NUMA (DRAM → L3 → L2 → L1 → CPU)" kullandığını, bu yüzden doğrusallaştırılabilir olmadıklarını ama sıralı tutarlılık sağladıklarını söylüyor. Bu ifadede üç sorun var:

- Sayılan şey bir **önbellek hiyerarşisidir**, NUMA değil. NUMA, belleğin işlemciler arasında bölünüp yerel ve uzak erişim sürelerinin farklı olmasıdır (M10.1).
- Önbellekler (MESI gibi protokollerle) **eşevrelidir**. Slayt 117 ve 132'de önbelleklerdeki "tutarsız kopyalar" olarak anlatılan sorunun asıl kaynağı başkadır: çekirdeklerin **yazma tamponları**, sırasız yürütme ve derleyici eniyilemeleri.
- Donanım sıralı tutarlı **değildir**. x86'nın modeli (TSO) bir yazmanın kendi tamponundayken sonraki bir okumanın önüne geçmesine izin verir (Şekil 2). ARM ve POWER daha da gevşektir.

Sıralı tutarlılığı sağlayan **dildir**. C/C++11'den beri, veri yarışı içermeyen bir program sıralı tutarlıymış gibi yürür (SC-DRF; aşağıda). Derleyici bunun için gerektiğinde çitler (fence) ve sıralı komutlar ekler.
:::

<figure class="fig wide"><svg viewBox="0 0 640 240" role="img" aria-label="Yazma tamponu litmus testi: sıralı tutarlılıkta ve x86-TSO modelinde olası sonuçlar"><text x="20" y="24" class="sm">başlangıç x = y = 0</text><rect x="20" y="34" width="180" height="56" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="32" y="56" class="sm">T1:  x = 1;</text><text x="32" y="78" class="sm">       r1 = y;</text><rect x="220" y="34" width="180" height="56" class="soft-b" stroke="currentColor" stroke-width="0.8"/><text x="232" y="56" class="sm">T2:  y = 1;</text><text x="232" y="78" class="sm">       r2 = x;</text><text x="20" y="122" class="sm">(r1, r2)</text><text x="110" y="122" class="sm">SC</text><text x="230" y="122" class="sm">x86-TSO</text><text x="20" y="146" class="sm">(0, 0)</text><text x="110" y="146" class="sm">olamaz</text><text x="230" y="146" class="tp">mümkün</text><text x="20" y="168" class="sm">(0, 1)</text><text x="110" y="168" class="tp">mümkün</text><text x="230" y="168" class="tp">mümkün</text><text x="20" y="190" class="sm">(1, 0)</text><text x="110" y="190" class="tp">mümkün</text><text x="230" y="190" class="tp">mümkün</text><text x="20" y="212" class="sm">(1, 1)</text><text x="110" y="212" class="tp">mümkün</text><text x="230" y="212" class="tp">mümkün</text><text x="430" y="56" class="sm">SC: 3 sonuç (program sırasını</text><text x="430" y="74" class="sm">koruyan 6 iç içe geçme).</text><text x="430" y="104" class="sm">TSO: 4 sonuç; (0, 0) da</text><text x="430" y="122" class="sm">mümkün: iki yazma da henüz</text><text x="430" y="140" class="sm">tamponda iken iki okuma</text><text x="430" y="158" class="sm">bellekteki eski sıfırları görür.</text><text x="430" y="188" class="sm">std::atomic (seq_cst) bu sonucu</text><text x="430" y="206" class="sm">yasaklar: derleyici bir çit</text><text x="430" y="224" class="sm">(mfence ya da xchg) ekler.</text></svg><figcaption>Şekil 2 — Donanım sıralı tutarlı değildir: x86'da bile her çekirdeğin yazma tamponu (0, 0) sonucuna izin verir; SC ancak veri yarışı olmayan ve seq_cst atomikler kullanan programlara dil tarafından sağlanır (sonuç kümeleri sayılarak bulundu).</figcaption></figure>

### Semaforlar {#semafor}

Semafor, Edsger Dijkstra'nın önerdiği, iki atomik işlemi destekleyen bir yazılım soyutlamasıdır: P (kilitle) ve V (aç). Uygulamalar şifreli P ve V yerine farklı adlar kullanır (kilitle–aç, al–bırak, bekle–sinyal), ama anlam semaforun türüne göre aynıdır.

**İkili semafor.** Durumu kilitli ya da açıktır.

- **P:** semafor açıksa kilitlenir ve çağrı döner. Kilitliyse thread bloklanır ve bekleyenler kuyruğu $Q$'ya konur.
- **V:** kuyruk boşsa semafor açılır. Değilse $Q$'dan bir thread çıkarılır ve devam eder; o thread'in P çağrısı tam bu anda döner.

Kuyruk FIFO ise semafor **güçlüdür** ve thread'lere geliş sırasıyla hizmet eder. Değilse **zayıftır**: bekleyen thread'ler arasında öncelik yoktur ve bir thread sonsuza dek bekleyebilir (**açlık**).

**Sayan semafor** (genel semafor). Durumu bir sayaç $C$'dir ve bir türden kaç kaynağın (mesaj, dosya tanıtıcısı, tampon) kullanılabilir olduğunu gösterir.

- **P:** $C$ azaltılır; yeni değer negatifse thread bloklanır.
- **V:** $C$ artırılır; yeni değer sıfır ya da negatifse bekleyen bir thread uyandırılır.

$C$ negatifken mutlak değeri bekleyen thread sayısıdır. Ders mantıksal bir gerçekleştirim de veriyor: `acquire`, `count > 0` olana kadar atomik bir denetim içinde döner (meşgul bekleme) ve sayacı azaltır; `release` atomik olarak artırır.

**Mutex ile ikili semafor farkı.** Mutex, ikili semaforun kısıtlı bir türüdür. Bir thread kilitlemeyi başarınca mutex'in "sahibi" olur ve onu yalnız o thread açabilir. Sahibi mutex'i ikinci kez kilitlerse, sıradan bir mutex'te bu bir kilitlenmedir. İkili semaforda bu kısıt yoktur: bir thread kilitler, başka bir thread açabilir. Bu, **olay bekleme** için yararlıdır. Bir sensör döngüsü veri gelince semaforu bırakır, işçi thread semaforu alıp veriyi işler ve tekrar almaya çalışarak bir sonraki olayı bekler. Mutex'ler varsayılan olarak açık başlatılır, semaforların başlangıç değeri ise açıkça verilir.

C++11 `mutex` sınıfını ekledi: `lock`, `unlock` ve bloklamayan, alınıp alınmadığını bool olarak döndüren `try_lock`. Sayan semafor C++20'de `std::counting_semaphore` sınıf şablonu olarak geldi; ikili semafor `std::binary_semaphore = counting_semaphore<1>`'dir. Derleyici desteği denetlenmelidir.

**Kritik bölgeler küçük olmalı.** Dersin dizi ortalaması örneğinde ilk sürüm, her elemanı tek bir küresel toplama kilitle ekler. Sonuç doğrudur, ama sıralı programdan **çok daha yavaştır**: kritik bölge thread'leri sıraya sokar ve kilidi alıp bırakmanın ek yükü her elemanda ödenir. Doğru sürüm, her thread'in kendi yerel toplamını yığınında hesaplayıp yalnız sonunda bir kez kilit almasıdır. Kritik bölgeler hem **sıklık** hem **süre** bakımından en aza indirilmelidir. Paylaşılan durum yerine thread'e özel kopya + birleştirme, paralel programlamanın en genel kalıbıdır.

**Tembel başlatma ve call_once.** Pahalı bir nesnenin yaratılmasını ilk gerektiği ana ertelemek (tembel başlatma) yüksek başarımlı yazılımda yaygın bir fikirdir. Paylaşılan bir nesne tembel başlatılacaksa hangi thread'in başlatacağı sorusu çıkar. Kolay yol, her kullanımdan önce bir mutex alıp işaretçi boşsa nesneyi yaratmaktır. Doğrudur, ama her çağrıda pahalı bir kilit–aç dizisi gerektirir. C++11 bunun için `call_once` ve `once_flag` getirdi: yalnız ilk çağrı fonksiyonu yürütür, diğerleri onun bitmesini bekleyip geçer. C++11'den beri fonksiyon içi `static` yerel değişkenlerin başlatılması da thread güvenlidir ("sihirli statikler").

**Özyinelemeli kilitler.** C++11, programlama hatası sayılabilecek bir şeye izin verir: aynı thread bir mutex'i birden çok kez kilitleyebilir. Böyle mutex'e **özyinelemeli** denir; açılması için aynı sayıda `unlock` gerekir. `mutex` özyinelemeli değildir. Örneğin her çağrısında aynı kilidi alan özyinelemeli bir fonksiyon için `recursive_mutex` kullanılır.

**Semaforun üç kullanımı:**

1. **Kilit olarak:** ikili semafor ya da mutex.
2. **Kaynak sayacı olarak:** genel semafor.
3. **Sinyalleşme için:** ilke olarak ikili ya da sayan semafor olabilir, mutex olamaz. Genel kural olarak ikili semafordan kaçınılır, çünkü birden çok sinyali biriktiremez.

**Yanlış gidebilecekler:**

- bir kaynağı alıp bırakmayı unutmak;
- bir mutex'i bir thread'de kilitleyip başka bir thread'de açmak (çoğu gerçekleştirimde tanımsız);
- hiç alınmamış bir kaynağı bırakmak;
- bir kaynağı kilitlemeden kullanmak. Salt okuma bile kilit gerektirebilir: o anda başka bir thread yazıyorsa.

### Kilitlenme, livelock ve açlık {#kilitlenme}

**Kilitlenme** (deadlock), iki ya da daha çok thread'in, ellerindeki kaynakları bırakmadan başka kaynaklara erişmek için sonsuza dek beklediği durumdur. P1 ve P2 süreçlerinin her biri R1 ve R2'nin ikisine de ihtiyaç duysun. İşletim sistemi R1'i P2'ye, R2'yi P1'e vermiş olsun. Her biri ötekinin kaynağını bekler ve hiçbiri işlevini yapmadan elindekini bırakmaz. İkisi de ilerlemez, uyur ya da bekler: CPU kullanımı düşüktür. Dersin örneğinde `thread1` önce `mutex1`'i, `thread2` önce `mutex2`'yi alır; ikisi de 100 ms uyuduktan sonra ötekini ister.

:::türetim Kilitlenmenin dört koşulu ve kilit sıralaması (Coffman ve ark. 1971)
Kilitlenme ancak dört koşul **birlikte** sağlanırsa oluşur:

1. **Karşılıklı dışlama:** bir kaynak aynı anda tek bir thread'dedir.
2. **Tut ve bekle:** bir thread elindekini tutarken başka bir kaynak bekler.
3. **Önalım yok:** bir kaynak sahibinden zorla alınamaz.
4. **Döngüsel bekleme:** $T_1$, $T_2$'nin tuttuğunu; $T_2$, $T_3$'ünkünü; …; $T_k$ de $T_1$'inkini bekler.

Birini kırmak kilitlenmeyi imkânsız kılar. En pratik olanı döngüsel beklemeyi kırmaktır. Bütün kilitlere küresel bir sıra ($L_1<L_2<\dots$) verilir ve her thread birden çok kilidi **yalnız artan sırada** alır.

Kanıt şudur. Bir döngü olsaydı, $T_i$'nin tuttuğu ve $T_{i-1}$'in beklediği kilidi $\ell_i$ ile gösterelim. Döngüdeki her thread elindeki kilitten büyük bir kilidi bekler: $\ell_1<\ell_2<\dots<\ell_k<\ell_1$. Bu bir çelişkidir.

C++17'nin `std::scoped_lock(m1, m2)`'si birden çok mutex'i kilitlenmesiz bir algoritmayla birlikte alır. `try_lock` + bırakma ise "tut ve bekle"yi kırar, ama livelock'a açıktır.
:::

**Livelock.** Bir thread çoğu zaman başka bir thread'in eylemine tepki verir. Öteki thread'in eylemi de bir tepkiyse livelock oluşabilir. Kilitlenmede olduğu gibi thread'ler ilerleyemez, ama **bloklu değildirler**: birbirlerine tepki vermekle meşguldürler. Gerçek hayattaki örnek, dar bir koridorda karşılaşan iki kişidir. İkisi de nazikçe kenara çekilip ötekine yol vermek ister, ama aynı anda aynı yöne kaydıkları için ilerleyemezler. Livelock'ta CPU, hesap yerine kilit mekanizmasına harcanır. Dersin örneğinde iki thread `try_lock` ile ilk kilidi alır, ikincisini alamazsa ilkini bırakıp yeniden dener ve sürekli "alamadım, yeniden deniyorum" yazar.

**Açlık** (starvation). Bir thread çalışmaya devam edebilecek durumdayken yürütmesi süresiz ertelenir. Genellikle önceliklerle ya da zamanlamada veya kaynak erişiminde adalet eksikliğiyle ilişkilidir. P1, P2 ve P3 bir kaynağı periyodik istesin. İşletim sistemi kaynağı dönüşümlü olarak P1 ve P3'e verirse, kilitlenme olmadığı hâlde P2 hiç erişemeyebilir. Dersin örneğinde yüksek öncelikli thread kaynağı sık alır; `try_lock` kullanan düşük öncelikli thread neredeyse hiç alamaz.

Aşağıdaki deney yemek yiyen filozofları simüle ediyor: halkadaki her thread sol ve sağ kilidini tutarak "yer".

- **Önce sol, sonra sağ:** süreler tam eşzamanlıyken 10. adımda kilitleniyor (döngüsel bekleme). Süreler rastgele saparsa kilitlenme olasılıksaldır.
- **Küresel sıra:** hiç kilitlenmiyor.
- **`try_lock` + sabit geri çekilme:** eşzamanlı durumda livelock'a giriyor; zamanın %99.5'i boşa gidiyor ve hiç yemek yenmiyor.
- **`try_lock` + rastgele geri çekilme:** simetriyi kırıyor.

::lab l2

### Üretici–tüketici {#uretici}

Semafor kullanımlarının her biri için bir kalıp vardır. Üretici–tüketici problemi, iki tür thread'in paylaşılan bir **sınırlı tampon** üzerinden eşzamanlanmasıdır. Tampon, üreticilerin yaptığı kaynakları tüketiciler isteyene kadar tutar; en genel biçimiyle $n$ üretici ve $m$ tüketici vardır. Gereksinimler şunlardır:

- Bir üretici tampona öğe koyarken `in` indeksini kilitlemelidir; tüketici öğe alırken `out` indeksini kilitlemelidir.
- Tampon boşsa tüketiciler beklemelidir.
- Tampon doluysa üreticiler beklemelidir.

Bunun için iki ikili semafor ya da mutex (`in` ve `out` için) ve iki sayan semafor (`slotsAvail` başlangıçta $B$, `resAvail` başlangıçta 0) gerekir. Tampon daireseldir: üreticiler ve tüketiciler farklı konumlarda çalışır.

```text
Üretici: item = produce(); slotsAvail.acquire(); l1.lock(); buffer[in] = item; in = (in+1) % B; l1.unlock(); resAvail.release();
Tüketici: resAvail.acquire(); l2.lock(); item = buffer[out]; out = (out+1) % B; l2.unlock(); slotsAvail.release(); consume(item);
```

**Eşzamanlamasız sürüm** tek üretici ve tek tüketiciyle çalışabilir. En önemli sorunu meşgul beklemedir: boş `while` döngüleri CPU yakar. Semafor, kaynak gelene kadar thread'i bloklar; döngüler `acquire` çağrılarına dönüşür ve CPU zamanı kazanılır.

Tasarım üzerine dört soru öğreticidir:

1. **`resCount` sayacına ne oldu?** Tamamen kalktı, çünkü hizmet ettiği meşgul bekleme döngüleri artık yok. Kalsaydı bir mutex gerekirdi.
2. **İki sayan semafor bire indirilebilir mi?** Sıralı bir programda her zaman `resAvail + slotsAvail = B` olurdu. Ama bir ya da iki thread tamponla etkileşirken bu toplam $B-1$, hatta $B-2$ olabilir. Daha önemlisi, iki semafor **sinyal** işlevi de görür: yer bekleyen üreticileri ve öğe bekleyen tüketicileri ayrı ayrı uyandırırlar.
3. **İki mutex bire indirilebilir mi?** Mümkün ama verimsizdir: bir üretici, bir tüketicinin `out`'u güncellemesini beklemek zorunda kalır. Tek üretici varsa `l1`, tek tüketici varsa `l2` tamamen kaldırılabilir.
4. **Kilit, öğe tampona yazılmadan bırakılabilir mi?** Konum geçici bir değişkende tutulup indeks güncellenir, kilit bırakılır, sonra öğe yazılır. Kritik bölge kısalır, ama ince bir hata doğar. Üretici 1 kilidi bıraktıktan hemen sonra, öğeyi koymadan kesilsin. Üretici 2 bir sonraki konuma öğe koyup `resAvail.release()` yapar. Uyanan tüketici, ayrılmış ama henüz yazılmamış ilk konumu okur.

Tamponu, yöntemleri kendi içinde kilitlenen bir sınıf yapmak ("thread güvenli tampon") eşzamanlama sorununu tampona taşır ve thread'leri eşzamanlılıktan habersiz bırakır. Bu genellikle iyi bir tasarım uygulamasıdır.

**Sonlandırma.** Paylaşımlı ve dağıtık uygulamalarda önemli bir sorun programın sonlandırılmasıdır. Sonlandırma koşulu genellikle yerel olarak değerlendirilir ve öteki thread'lere yayılmalıdır. Örneğin bir parola kırma programında eşleşen bir hash bulunur, ya da arayüz thread'i Çıkış düğmesine basıldığını görür. İki şart vardır: sinyal bütün thread'lere ulaşmalı ve thread'ler onu makul bir sürede fark etmelidir. İki yol vardır:

1. **Paylaşılan bayrak.** Basit görünür ama tuzakları vardır. Bayrak bir `std::atomic<bool>` olmalıdır. Ayrıca bir semaforda bloklu bekleyen bir thread bayrağı hiç göremez: tüketiciler boş tamponda `resAvail.acquire()`'da uyurken üreticiler biterse sonsuza dek uyurlar. Ders bunu iki durumda ele alır: yineleme sayısı önceden biliniyorsa her thread kaç öğe işleyeceğini bilir; bilinmiyorsa bayrağı koyan thread bekleyenleri ayrıca uyandırmalıdır.
2. **Mesajlarla sonlandırma.** Dağıtık belleğe de uzanan zarif bir yoldur. Tamponda dolaşan öğeler örtük mesajlardır. Uygulamada geçersiz özel bir öğe, **zehirli hap** (poison pill) programın sonunu bildirir: yalnız pozitif sayı bekleyen bir uygulamada negatif bir sayı, nesne işaretçisi bekleyen tüketicilerde boş işaretçi. Her tüketici için bir hap konur; hapı alan tüketici çıkar.

:::uyarı Kaynak düzeltmesi (slayt 118): volatile thread'ler arası bayrak değildir
Slayt sonlandırma bayrağı için "ya atomik bir tür ya da derleyicinin kopyayı önbelleğe almasını önleyen `volatile`" önerisinde bulunuyor. C ve C++'ta `volatile` thread'ler arası eşzamanlama sağlamaz. Derleyicinin erişimi kaldırmasını önler, ama atomiklik, sıralama ya da görünürlük güvencesi vermez. Bir thread'in yazıp ötekinin okuduğu `volatile bool` hâlâ bir **veri yarışıdır**, yani tanımsız davranıştır. `volatile` bellek eşlemeli donanım yazmaçları içindir (Java'nın `volatile`'ı ise farklıdır ve bu işi görür). Doğrusu `std::atomic<bool>` kullanmaktır; varsayılan seq_cst sıralaması bayrağın ve ondan önceki yazmaların görünmesini güvence eder.
:::

Aşağıdaki deney 2 üretici, 3 tüketici ve 8 yuvalı bir tamponla 200 öğe işliyor. Tüketiciler üreticilerden hızlı. Zehirli hapla bütün tüketiciler çıkıyor. Bayrakla bir tüketici semaforda takılı kalıyor. Meşgul beklemede bayrak görülüyor, ama bunun bedeli toplam sürenin 1.8 katı çekirdek zamanının boşa gitmesi.

::lab l3

### Atomik türler ve CAS {#atomik}

C++11 ve sonrası bir `atomic` sınıf şablonu ve çeşitli özelleştirmeler içerir. Atomik türler açık bir kilit kullanmadan veri yarışından arınmış erişim sunar ve kilitsiz eşzamanlı veri yapılarının yapı taşıdır. Hedef mimarinin desteğine göre perde arkasında kilit kullanılabilir; `is_lock_free()` bunu sorar. Atomik yöntemler şunlardır:

- `store` / `load`: değeri yazar / okur.
- `fetch_add` / `fetch_sub`: ekler / çıkarır ve önceki değeri döndürür.
- `fetch_and` / `fetch_or` / `fetch_xor`: bit işlemleri yapar ve önceki değeri döndürür.
- `exchange`: değeri değiştirir ve öncekini döndürür.
- `compare_exchange_weak` / `compare_exchange_strong`: atomik değer beklenene eşitse yenisini yazar, değilse gerçek değeri beklenen değişkene yükler.
    - **Zayıf** sürüm sahte (spurious) başarısız olabilir: değer eşit olduğu hâlde, gerçekleştirimden dolayı başarısız döner. Zaten bir döngü içinde kullanılıyorsa bazı platformlarda daha hızlıdır.
    - **Güçlü** sürüm sahte başarısızlıkta yeniden dener.

Ayrıca `++`, `--`, `+=`, `-=`, `&=`, `|=`, `^=` işleçleri vardır, ama her türde değil. Örneğin atomik işaretçide `&=` anlamsızdır ve yoktur.

:::uyarı Kaynak düzeltmesi (slayt 121): std::atomic yalnız temel türler için değildir
Slayt `atomic<FooClass>`'ın "derlenmediğini", şablonun "yalnız standardın belirttiği temel türler için" çalıştığını söylüyor. `std::atomic<T>` **her trivially copyable** $T$ için tanımlıdır: kopya yapıcısı, kopya atama işleci ve yıkıcısı önemsiz olan her sınıf, örneğin iki `int`'lik bir yapı. Böyle bir tür donanımın atomik genişliğinden büyükse gerçekleştirim perde arkasında bir kilit kullanır (`is_lock_free()` false) ve `fetch_add` gibi aritmetik yöntemler sunulmaz. Derlenmeyen şey, trivially copyable olmayan sınıflardır (örneğin içinde `std::string` olan). Kayan noktalı `fetch_add` ve `std::atomic_ref` C++20'de gelmiştir.
:::

**CAS ve LL/SC.** Çok düzeyli önbelleklerin ve çok thread'li yürütmenin varlığında atomik bellek işlemleri, değişiklik işlenmeden önce yapılan bir denetime dayanır. İki yaklaşım vardır:

- **Karşılaştır ve değiştir** (compare-and-swap, CAS): bir bellek konumunun içeriğini bir değerle karşılaştıran ve eşleşirse yeni değeri yazan tek bir komuttur; her durumda okunan değer döner. "Karşılaştır ve ayarla" türü ise yazmanın olup olmadığını bool olarak döndürür. x86-64 bunu 8, 16, 32 ve 64 bitlik `CMPXCHG` ile yapar.
- **Yüklemeye bağla / koşullu sakla** (load-link/store-conditional, LL/SC): RISC mimarileri (PowerPC, ARM) iki komut kullanır. LL, konumun içeriğini döndürür. SC aynı konuma yeni değeri yazmaya çalışır; LL'den beri o konuma başka bir değişiklik yapılmışsa ya da arada bağlam değiştiyse başarısız olur. Zayıf CAS'ın sahte başarısızlığı buradan gelir: SC başka nedenlerle de başarısız olabilir.

CAS'ın bilinen tuzağı **ABA sorunudur**. Bir thread A değerini okur, başka thread'ler değeri B'ye sonra yine A'ya çevirir; CAS "değişmemiş" sanıp başarılı olur. Kilitsiz bir yığında bu, silinip yeniden kullanılmış bir düğümün işaretçisi olabilir. LL/SC değere değil yazmaya baktığı için bu sorunu yaşamaz. CAS'ta sayaçlı etiketli işaretçiler kullanılır.

Atomik türler kilit kurmak, kilitlerin yerine geçmek ya da onları tamamlamak için kullanılabilir. Ders, `atomic<bool>` ile bir kilit kuruyor: `compare_exchange_weak(tmp, true)` başarılı olana kadar döner, başarısızlıkta `this_thread::yield()` ile CPU'yu bırakır. `tmp` her turda `false`'a yeniden atanmalıdır, çünkü başarısız karşılaştırma ona gerçek değeri (true) yazar. Çıkış `lock.store(false)`'tur.

### Bellek sıralaması {#siralama}

Atomik yöntemlerin isteğe bağlı bellek sıralaması parametreleri vardır. Bellek sıralaması thread'ler arası eşgüdümle, yani bir thread'deki bellek işlemlerinin öteki thread'lerce nasıl algılandığıyla ilgilidir. Tek bir thread'in kendi işlemlerine etkisi yoktur: bir thread kendi değişikliklerini her zaman görür. C++11 altı sıralama tanımlar:

1. **`memory_order_relaxed`:** yalnız o değişken üzerindeki işlemin bölünmezliği. Öteki okuma ve yazmalara hiçbir sıralama kısıtı koymaz. Her değişkenin bir değişiklik sırası vardır; farklı thread'ler farklı değişkenlerin güncellemelerini farklı sıralarda görebilir. Yalnız sıra önemsizken uygundur, örneğin bir istatistik sayacı.
2. **`memory_order_acquire`:** bir yüklemede kullanılır. Bu yüklemeden **sonraki** okuma ve yazmalar onun önüne geçemez.
3. **`memory_order_release`:** bir saklamada kullanılır. Bu saklamadan **önceki** okuma ve yazmalar onun arkasına geçemez.
4. **`memory_order_acq_rel`:** oku–değiştir–yaz işlemleri (`compare_exchange`, `fetch_add`) içindir; okuma kısmında acquire, yazma kısmında release uygular.
5. **`memory_order_consume`:** acquire'ın veri bağımlılığına sınırlanmış hafif sürümüdür. Pratikte bütün derleyiciler onu acquire'a yükseltir ve standart kullanımını önermez.
6. **`memory_order_seq_cst`:** acquire–release'e ek olarak bütün seq_cst işlemleri için **tek bir küresel sıra** güvence eder. En katıdır ve varsayılandır.

**Acquire–release eşleşmesi.** Thread A verisini hazırlayıp bir bayrağa release ile yazsın; thread B bayrağı acquire ile okuyup yazılan değeri görsün. O zaman A'nın release'ten önceki **bütün** yazmaları B'de acquire'dan sonra görünür. Mutex'in kilitlenmesi acquire, açılması release'tir. Bir kilit içinde yapılan bütün değişikliklerin bir sonraki kilit sahibine görünmesi bu yüzdendir.

:::uyarı Kaynak düzeltmesi (slayt 50, 134 ve 138): acquire–release ne yapar, kime gerekir?
Slayt 50 ve 134 acquire'ı "o atomik değişkene diğer thread'lerin bütün yazmaları görünür olur" diye tanımlıyor. Bir atomik değişkenin kendi değeri için bu zaten her sıralamada geçerlidir (değişiklik sırası). Acquire'ın asıl işi **başka** değişkenlerle ilgilidir: release eden thread'in o noktadan önceki yazmalarını acquire edene taşır. Mesaj geçirme (veri + bayrak) ve kilitler buna dayanır.

Slayt 138 acquire–release'in "eski ARM ve PowerPC için eklendiğini, modern işlemcilerde hiç gerekmeyeceğini" söylüyor. Tersine, acquire–release her mimaride kilitlerin ve tek üretici–tek tüketici kuyruklarının standart aracıdır. Seq_cst varsayılandır ve doğruluk için güvenli seçimdir. Ama ARM ve POWER gibi gevşek modellerde fazladan çit ister: x86'da seq_cst bir saklama `mfence` ya da `xchg` gerektirir, release saklama sıradan bir `mov`'dur. Önerilen yol önce seq_cst ile doğru yazmak, gevşetmeyi ancak ölçülmüş bir darboğazda ve dikkatli bir akıl yürütmeyle yapmaktır.
:::

Ders üç çalışma kipini özetliyor:

- **Gevşek** (relaxed): "çok emin değilseniz kullanmayın".
- **Acquire–release:** thread çiftleri arasında yayılım sağlar.
- **Sıralı tutarlı:** varsayılandır.

Dersin gevşek kip örneğindeki doğrulama (assertion) x86-64'te asla tetiklenmez, çünkü x86'nın güçlü sıralaması yeniden sıralamayı donanımda önler. Tetiklenmesi için gevşek modelli bir mimari (ARM, POWER) gerekir. Gevşek atomiklerle yazılmış hatalı kod bu yüzden x86'da yıllarca "çalışabilir".

**SC-DRF.** C/C++11 ile çok thread'li programlama desteğinin yanında derleyici geliştiricileri ve yazılımcılar bir bellek modelinde uzlaştı: **veri yarışı içermeyen bir program sıralı tutarlı yürür**. Kural, çok thread'li programlar hakkında akıl yürütmeyi çok kolaylaştırır: veri yarışınız yoksa kodunuz yazıldığı sırayla yürüyormuş gibi davranır. Bu model yüksek başarımlı derleyiciler için de uygundur, çünkü veri yarışı olmayan kodu thread'leri hiç düşünmeden eniyileyebilirler. Gerçekleştirimde atomik okumalar acquire yükleme, atomik yazmalar release saklama olarak (seq_cst için gerektiğinde ek çitlerle) derlenir.

### Monitörler ve koşul değişkenleri {#monitor}

Semaforlar büyük esneklik sunar ve doğru kullanılırsa yüksek başarım verir. Ama doğru kullanmaları zordur: çok alt düzey bir soyutlamadırlar ve nesne yönelimli programlama çağında yerlerini yadırgatırlar. Atomikler ise yürekli olmayanlara göre değildir. Daha güvenli bir seçenek **monitördür**: birden çok thread'in güvenle erişmesi için tasarlanmış bir nesne ya da modül. Monitörün yöntemlerinden herhangi birini aynı anda yalnız bir thread yürütebilir. Bir thread'i bloklamak ya da devam ettirmek için **koşul değişkenleri** kullanılır. Bir koşul değişkeninin iki işlemi vardır:

- **Bekle:** thread koşulsuz bloklanır, monitörün kilidini bırakır ve koşul değişkeninin kuyruğuna girer.
- **Sinyal:** bekleyen bir thread varsa uyandırılır; yoksa sinyal **kaybolur**.

Semafordan farklar şunlardır. Bir semaforda P ancak sayaç sıfırsa bloklar; release ise her zaman kaydedilir ve sonraki bir acquire'a sayılır. Koşul değişkeninde wait her zaman bloklar, kimse beklemiyorsa sinyal hiçbir iz bırakmaz. Bir koşul değişkeni her zaman bir **koşulla**, yani beklemenin ne zaman biteceğini anlatan bir kurallar kümesiyle ilişkilidir. Monitörün semafora üstünlüğü budur: herhangi bir koşul, programcının kendi mantığıyla açıkça ifade edilir.

Sinyalden sonra uyanan thread'in ne zaman çalışacağı iki türlü olabilir:

- **Hoare monitörü** (1974): uyanan thread hemen çalışır ve sinyal veren askıya alınır. Monitörde aynı anda iki etkin thread olamayacağı için bu zorunludur.
- **Lampson–Redell (Mesa) monitörü:** uyanan thread, sinyal veren monitörden çıkana kadar bekler. Bu arada beklediği koşul yeniden değişmiş olabilir; bu yüzden bekleme `if` ile değil **`while`** ile denetlenir. Bunun iki yararı vardır: zaman aşımlı bekleme ve bir koşulu bekleyen bütün thread'leri uyandırma. Uyanan thread koşulu yeniden değerlendirdiği için ikisi de güvenlidir.

Farkı belirtmek için sinyal işlemine `notify`, hepsine sinyale `notify_all` denir. C++11, Qt ve Java'daki monitör gerçekleştirimlerinin çoğu Lampson–Redell'i izler, çünkü ek işlevselliği daha basit bir gerçekleştirimle gelir. Bekleme ayrıca **sahte uyanmalara** (spurious wakeup) karşı da `while` ister: işletim sistemi bir thread'i sinyalsiz uyandırabilir.

C++11 `condition_variable` şu yöntemleri sunar:

- `wait`: thread'i bloklar. Monitör kilidini ayrıca açıp kapamaya gerek kalmasın diye kilit parametre olarak verilir.
- `wait_for`: belirli bir süre ya da sinyal gelene kadar bloklar.
- `wait_until`: belirli bir zaman noktasına kadar bloklar.
- `notify_one`: bekleyenlerden birini uyandırır.
- `notify_all`: hepsini uyandırır; monitöre birer birer girerler.

`wait(lock, pred)` aşırı yüklemesi koşul döngüsünü içerir: `while (!pred()) wait(lock);`.

`wait`'e verilen kilit bir **`unique_lock`** olmalıdır. Bu, bir mutex'in tek sahipliğini yöneten genel amaçlı bir sarmalayıcıdır. Yapıcısında kilitler, yıkıcısında (istisna fırlatılsa bile) açar ve ömrü boyunca kilitleyip açmaya izin verir. `wait` onu içeride açıp uyanınca yeniden kilitler. Bloklamayan yöntemlerde daha basit olan `lock_guard` yeter: yapıcıda kilitler, yıkıcıda açar, arada açılamaz.

Banka hesabı monitörü şöyledir:

```cpp
class AccountMonitor {
  mutex m; condition_variable insufficientFunds; double balance = 0;
public:
  void withdraw(double s) { unique_lock<mutex> ml(m); while (balance < s) insufficientFunds.wait(ml); balance -= s; }
  void deposit(double s)  { lock_guard<mutex> ml(m); balance += s; insufficientFunds.notify_all(); }
};
```

`withdraw` bakiye yetene kadar bekler; `deposit` bekleyenlerin hepsini uyandırır. Uyananlar `while` ile bakiyeyi yeniden denetler. Bu yüzden tek bir yatırma büyük bir çekmeyi karşılamaya yetmezse çekme yeniden uyur.

### Eşzamansız görevler {#async}

C++11 bir fonksiyonu "arka planda" çalıştırmak için `async` şablon fonksiyonunu sunar. Anahtar sözcük "istenir"dir: varsayılan politikada, yeni bir thread mi açılacağına yoksa fonksiyonun aynı thread'de mi çalışacağına kütüphane karar verir. Politika açıkça seçilebilir:

- `launch::async`: yeni bir thread zorunludur.
- `launch::deferred`: thread açılmaz. Çağıran sonucu `get()` ile istediğinde fonksiyon o anda çalışır (tembel değerlendirme).

`async`'in thread yapıcısından asıl farkı, fonksiyonun sonucunun çağırana **açıkça** dönmesidir. Daha önce bunu functor'larla ya da başvuru parametreleriyle örtük yapıyorduk. Dönüş bir **`future`** nesnesidir. Ders, ortalama programını `async` ile yeniden yazıyor: her parça için `f[i] = async(launch::async, partialSum, start, end)`, sonra `res += f[i].get()`. Ne mutex ne açık join gerekir.

`future` yöntemleri şunlardır:

- `get`: sonucu bekler ve döndürür.
- `wait`: sonuç hazır olana kadar bekler.
- `wait_for` / `wait_until`: zaman aşımlı bekler ve `deferred` (henüz çağrılmadı), `ready` ya da `timeout` döndürür.

Örneğin uzak bir sunucuya bağlantıyı en çok 1 saniye beklemek için `if (f.wait_for(1s) == future_status::ready) sockfd = f.get();` yazılır.

**`packaged_task`** herhangi bir çağrılabilir hedefi (fonksiyon, functor, lambda) saklanıp taşınabilen bir nesneye sarar. `async`'ten farkı, yalnız bir kap olmasıdır: yaratmak bir thread açmaz, hedefi çalıştırmaz. Çalıştırmak için açıkça çağrılır (`pt(1.0, 2.0)`) ya da bir thread'e taşınır (`thread t(move(pt), 1, 2)`); sonuç önceden alınmış `pt.get_future()` ile okunur. Parametreler `bind` ya da bir lambda ile görevin içine saklanabilir. Thread havuzlarının görev kuyrukları genellikle `packaged_task` tutar.

### Thread havuzları ve fiber'ler {#havuz}

Şimdiye kadar her iş için bir çocuk thread açıkça (`thread`) ya da örtük (`async`) başlatıldı. Bu dinamik yaklaşım çoğu senaryoya yeter, ama thread yaratma ve yok etme sık yapılırsa başarımı tüketir. Saniyede binlerce istek alan bir internet sunucusu buna örnektir. Thread yaratmak süreç yaratmaktan ucuzdur ama bedava değildir: bellek bölgeleri ayrılıp başlatılır, çekirdek yapıları güncellenir. Makinenin kapasitesine bakmadan thread yaratmak da sorunludur. İşlemcileri dengelemek için 10 000 thread isteyen bir algoritma, yaratma ve yok etme ek yükü yüzünden büyük olasılıkla başarısız olur. İşletim sistemi bu kadar thread'e zaten izin vermeyebilir.

Seçenek, bir kuyruktan iş alan sabit bir thread kümesidir: **thread havuzu**.

Artıları:

- Her görev için başlatma ek yükü yoktur; yalnız havuz bir kez başlatılır.
- Thread sayısı işe göre ayarlanabilir. G/Ç bağımlı işler CPU'yu doldurmak için daha çok thread isteyebilir; CPU bağımlı işler mantıksal çekirdek başına bir thread'le yetinir.
- Yük dengeleme için binlerce görev üretilebilir; işletim sistemi sınırları engel olmaz.
- Sistem kaynak tüketimi (bellek, çekirdek yapıları) en azdır. Önbellek kullanımı da iyileşir.

Eksileri:

- Görevler bir kaba (ideal olarak bir kuyruğa) konup çıkarılabilecek biçimde paketlenmelidir. Sonuçları toplamanın bir yolu da gerekir.
- Görev önceliğini zorlamanın kolay bir yolu yoktur. Bitmesini beklemesi gereken alt görevler üreten görevler de kolay değildir; bunlar fiber'lerle ya da birden çok havuzla yapılabilir.
- İşbirlikçi görevler sorun olabilir. N üretici ve M tüketici görevi varsa havuz en az N + M thread büyüklüğünde olmalıdır. Aksi hâlde bloklanan üreticiler bütün thread'leri tutar ve tüketiciler hiç çalışamaz: havuz düzeyinde bir kilitlenme.

**Fiber'ler.** Thread'lerin ilk yıllarında işletim sistemi desteği yoktu ve üçüncü taraf kütüphaneler kullanılıyordu. Geçiş **işbirlikçiydi**: çalışan thread CPU'yu bırakmak için kütüphaneyi çağırır, kütüphane başka bir thread'i seçer. Bu, tek bir süreç bağlamında olur; bu yüzden **işbirlikçi çok iş parçacıklılık** ya da **kullanıcı düzeyi** thread denir. Bugünkü işletim sistemlerinin hepsi thread'leri sistem düzeyinde destekler: zamanlayıcı thread'lerden haberdardır (**çekirdek düzeyi** thread'ler).

Kullanıcı düzeyi thread'lerin artıları şunlardır:

- Yaratma, haberleşme ve geçiş maliyetleri çekirdek thread'lerinin küçük bir kesridir. "Hafif thread", "yeşil thread" ve "mikro thread" adları buradan gelir.
- Yüzlerce, binlerce thread kolayca yaratılır.
- Özel zamanlayıcılar kullanılabilir.
- Taşınabilirlik daha kolaydır.

Eksileri şunlardır:

- Bir thread süreci bloklayan bir sistem çağrısı yaparsa **hepsi** bloklanır; bu yüzden bloklamayan G/Ç tercih edilir.
- Aynı anda yalnız biri çalıştığı için tek bir çekirdek kullanılır.

**Fiber**, ikisini birleştirir. Çekirdek thread'leriyle bütün çekirdekleri doldururuz; her thread de kendi içinde birden çok kullanıcı düzeyi thread (fiber) yaratıp her çekirdeğin ne yapacağını en düşük maliyetle ince ayarlar. Boost.Fiber kütüphanesi:

- **Yaşam döngüsü:** fiber'ler `join` edilir ya da ayrılır (`detach`).
- **Eşzamanlama:** fiber'ler arası eşzamanlama `boost::fibers::mutex` ve `condition_variable` ile yapılır. Bir fiber `std::mutex`'te bloklanırsa o thread'in **bütün** fiber'leri bloklanır. Bu yüzden thread içi fiber eşzamanlaması için Boost'unkiler, thread'ler arası eşzamanlama için `std` olanları kullanılır.
- **Uyutma ve geçiş:** `this_fiber::sleep_for` / `sleep_until` uyutur, `this_fiber::yield()` başka bir fiber'e geçer. `std::this_thread::yield()` ise bütün thread'i bırakır.
- **Zamanlama:** varsayılan zamanlama döngüseldir (round-robin). Diğer seçenekler şunlardır:
    - `work_stealing`: yerel kuyruk boşsa başka thread'lerin kuyruklarından fiber çalar.
    - `numa::work_stealing`: aynısını NUMA düğümlerini gözeterek yapar.
    - `shared_work`: fiber'ler bütün thread'lerce paylaşılır; NUMA için kötü bir seçimdir.
- **Havuz olarak kullanım:** thread ve fiber'leri birleştirerek, görevin bloklanması sorununu hafifleten bir havuz kurulabilir. Bir fiber kütüphane üzerinden bloklanırsa yalnız o fiber bekler, thread başka bir fiber'e geçer.

Fiber'lerin en iyi kullanımı, sık bağlam geçişinin gerektiği senaryolardır. Çekirdek thread'leri arasında geçiş bir sistem çağrısı ve çekirdek kipine girip çıkmak demektir; fiber'ler ise kullanıcı kodu içinde, birkaç yazmaç kaydederek geçer. Bugün C++20 eşyordamları (`co_await`) aynı fikrin dil düzeyindeki biçimidir.

:::uyarı Kaynak düzeltmeleri (fiber'ler ve akıllı işaretçiler)
- **Fiber'ler:** slayt fiber'leri "görece yeni bir C++ özelliği, çoğunlukla Windows için" diye tanıtıyor. Fiber'ler standart C++'ın parçası değildir. Boost.Fiber platformlar arası bir kütüphanedir; Windows'un ayrıca kendi işletim sistemi fiber API'si (`CreateFiber`) vardır. Standart C++'taki karşılık C++20 eşyordamlarıdır (yığınsız).
- **Akıllı işaretçiler:** `make_unique` ve `make_shared` vardır, ama slaytta geçen **`make_weak` yoktur**. `weak_ptr` bir `shared_ptr`'dan kurulur: `weak_ptr<T> w = sp;`.
- **Thread güvenli ve yeniden girilebilir:** slayt ikisini eş anlamlı sayıyor. Ama bir fonksiyon thread güvenli olup yeniden girilebilir olmayabilir (örneğin bir mutex kullanan fonksiyon, bir sinyal işleyicisinden yeniden çağrılırsa kilitlenir). Tersi de mümkündür.
:::

**Akıllı işaretçiler ve thread güvenliği.** C++'ın dinamik bellek yönetimi `new` ve `delete`'tir. Akıllı işaretçiler bunu otomatikleştirir; kapsam dışına çıkınca silinir ya da başvuru sayıları azalır (`#include <memory>`):

- **`unique_ptr`:** kopyalanamaz, tek sahiplidir.
- **`shared_ptr`:** kopyalanabilir, başvuru sayımlıdır. Her kopya sayıyı bir artırır, her yok etme bir azaltır; sayı sıfıra inince nesne silinir.
- **`weak_ptr`:** başvuru sayısını artırmadan bir `shared_ptr` nesnesini gözler.

Bir fonksiyon ya da nesne, birden çok thread aynı anda çağırdığında doğru davranıyorsa **thread güvenlidir**. `shared_ptr` thread güvenli midir? Başvuru **sayacı** atomik güncellenir: farklı thread'lerin aynı nesneyi gösteren farklı `shared_ptr` kopyalarını yaratıp yok etmesi güvenlidir. Ama aynı `shared_ptr` **nesnesine** birden çok thread'in yazması bir veri yarışıdır (C++20'de `atomic<shared_ptr<T>>` vardır). Gösterilen nesnenin kendisine erişim de ayrıca korunmalıdır.

### Çok thread'li programlarda hata ayıklama {#hata}

Çok thread'li uygulamalarda hata ayıklama, yalnız birden çok thread'i yürütebilen bir hata ayıklayıcıya sahip olmanın ötesine geçer. Çağdaş hata ayıklayıcıların çoğu (GDB ve DDD gibi ön yüzleri) thread'lerin ayrı ayrı yürütülmesini ve incelenmesini destekler; tek gereksinim çalıştırılabilir dosyanın hata ayıklama bilgisiyle derlenmesidir. Ama:

- Çok thread'li programlardaki hatalar genellikle seyrek ortaya çıkar, yalnız belirli zamanlama koşullarında.
- Hata ayıklayıcı yürütmeyi durdurup yavaşlatarak zamanlamaları bozar. Hata gözlenmeye çalışıldıkça kaybolabilir ("Heisenbug").

Hataları bulmak sonunda deneyim ve sezgi işidir, ama bazı kurallar vardır:

1. **Hiç hata yapmamaya çalış.** Koddan önce sağlam bir yazılım tasarımı kritiktir. Bu bölümdeki klasik problemler yalnız eğitsel değildir: gerçek hayattaki eşzamanlılık sorunlarının çoğu bunların örnekleri ya da birleşimleridir, ve çözümlerini kullanmak tekerleği yeniden icat etmekten doğan hataları önler.
2. **Günlük tut.** Uygulama bir tür günlük ya da iz geçmişi üretmelidir; bu, çalışma zamanı davranışı hakkında bilgi toplamayı sağlar. Birden çok thread'in aynı anda konsola ya da dosyaya yazması iç içe geçmiş, okunamaz çıktı verir. Thread başına ayrı tampon, ya da kimlik ve zaman damgalı tek bir kilitli günlük kullanılır.
3. **Thread sayısını azalt.** Çok sayıda thread çıktının yorumunu zorlaştırır. Thread sayısını 1'e indirmek zamanlamaya özgü olmayan hataları ortaya çıkarır; 2–3'e indirmek bilgi toplamanın ek yükünü azaltır.
4. **Eniyilemeyi kapat.** Eniyileyen bir derleyici deyimlerin yürütme sırasını değiştirebilir, hatta bazılarını atabilir. Eniyilenmiş bir çalıştırılabiliri adım adım izlemek "kaynak kodda olmayan sıçramalar" gibi şaşırtıcı sonuçlar verebilir. Nadiren hatayı eniyilemeler ortaya çıkarır: genellikle veri yarışı içeren kod, eniyileme açılınca bozulur (bir bayrak döngüsünün sonsuza dönmesi gibi). Bu durumda sorun derleyicide değil koddadır.
5. **Eniyilemeleri en sona bırak.** Önce tek thread'li sürümle aynı sonucu verdiğinden emin ol, sonra hızlandır.

Bugün bunlara iki güçlü araç eklenmelidir:

- **ThreadSanitizer** (`-fsanitize=thread`): veri yarışlarını çalışma sırasında, gerçekleşmeseler bile bulur.
- **Helgrind** (Valgrind): kilit sırası ihlallerini, yani olası kilitlenmeleri bildirir.

**Aynı mutex'i iki kez kilitlemek** mutex türüne bağlıdır. İşletim sistemleri genellikle üç tür sunar:

- **hızlı mutex:** aynı thread'de iki kez kilitlemek kilitlenmedir;
- **özyinelemeli mutex:** aynı sayıda açmak gerekir;
- **hata denetimli mutex:** ikinci kilitleme hemen bir hata koduyla döner.

Bazı işletim sistemleri yalnız hızlısını sunar.

**Donanım ve derleyici.** Bütün bu kavramlar (atomikler, mutex'ler, semaforlar, koşul değişkenleri) sonunda yazılımda ve donanımda gerçekleştirilir. Yazılım tarafında yük, çok iş parçacıklı API'lerde ve en önemlisi derleyicidedir. Donanım tarafında her şey sonunda atomikliği ve sıralamayı sağlayan belirli komutlara iner: x86'da `lock` önekli komutlar ve `mfence`, ARM'da LDREX/STREX ve `dmb`. SC-DRF modeli bu ikisini birbirine bağlayan sözleşmedir.

### Özet {#ozet}

- Süreçler ayrı adres uzayıdır, `fork()` ile yaratılır (çocuk 0, ebeveyn pid alır; yazınca kopyala). Thread'ler yalnız yığını ayrı tutar.
- `std::thread` ile yaratılır, `join` ya da `detach` edilir (birleştirilebilir thread'in yıkımı `terminate`'tir). Parametreler değerle geçer, `ref` gerekir. `jthread` otomatik join eder.
- Veri yarışı tanımsız davranıştır; yarış koşulu zamanlamaya bağlı hatadır. Biri öbürünü gerektirmez. Bölünmez olması gereken, oku–değiştir–yaz bütünüdür: 2 adımda 6 sıranın 4'ü, 3 adımda 20 sıranın 18'i yanlıştır.
- Kilit, mutex ve semafor (ikili/sayan, güçlü/zayıf) temel araçlardır. Kritik bölgeler kısa olmalıdır; thread'e özel kopya + birleştirme kullanılır. `call_once`, `recursive_mutex` ve semaforun üç kullanımı da bu araç takımındadır.
- Sıralı tutarlılık ve doğrusallaştırılabilirlik iki tutarlılık modelidir. Donanım SC değildir (TSO yazma tamponu); dil, SC-DRF ile SC verir.
- Kilitlenmenin dört koşulu vardır ve küresel kilit sırası döngüyü kırar. Livelock'ta thread'ler meşguldür ama ilerlemez; rastgele geri çekilme bunu çözer. Açlık adaletsizlikten doğar.
- Üretici–tüketici iki sayan semafor ve iki mutex'le kurulur. Sonlandırma için zehirli hap kullanılır; bayrak `atomic` olmalıdır, `volatile` olmaz.
- `std::atomic`, CAS ve LL/SC araçları ABA sorununu ve bellek sıralamalarını (relaxed, acquire/release, seq_cst) birlikte getirir.
- Monitör ve koşul değişkeni Lampson–Redell (Mesa) biçimindedir: bekleme `while` ile yapılır, `unique_lock` ve `lock_guard` kullanılır.
- `async`/`future`, `packaged_task`, thread havuzları ve fiber'ler daha üst düzey araçlardır. Hata ayıklamada günlük tutulur, thread sayısı azaltılır, ThreadSanitizer kullanılır.
