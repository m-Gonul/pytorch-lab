### Verimli temsiller: neden kısıt? {#verimli}

**Otomatik kodlayıcılar** (autoencoders), girdi verisinin **gizil temsil** ya da **kod** denen yoğun temsillerini hiçbir denetim olmadan (eğitim kümesi etiketsizken) öğrenebilen sinir ağlarıdır. Kodlar tipik olarak girdiden çok daha düşük boyutludur; bu yüzden otomatik kodlayıcılar boyut indirgemede, özellikle görselleştirmede kullanışlıdır. Ayrıca öznitelik dedektörü olarak çalışırlar ve derin sinir ağlarının denetimsiz ön eğitiminde kullanılabilirler. Bazı otomatik kodlayıcılar üretkendir (VAE, M5.1): eğitim verisine çok benzeyen yeni veriler üretebilirler.

Otomatik kodlayıcı girdisini çıktısında yeniden kurmayı öğrenir. Bu kulağa önemsiz gelir: özdeşlik fonksiyonunu öğrenmek ne işe yarar? İşe yaraması için ağ çeşitli biçimlerde **kısıtlanır**. Kodun boyutu sınırlanabilir, ya da girdiye gürültü eklenip ağdan orijinal girdiyi kurması istenebilir. Bu kısıtlar ağın girdiyi doğrudan çıktıya kopyalamasını engeller ve veriyi temsil etmenin verimli yollarını öğrenmeye zorlar. Kısacası kodlar, kısıtlar altında özdeşlik fonksiyonunu öğrenmenin **yan ürünleridir**.

**Hangi dizi daha kolay ezberlenir?** 40, 27, 25, 36, 81, 57, 10, 73, 19, 68 mi; yoksa 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14 mü? İlk bakışta ilki daha kısa olduğu için daha kolay görünür. Ama dikkatli bakınca ikinci dizinin 50'den 14'e inen çift sayılar olduğu görülür. Deseni fark edince ikinci dizi çok daha kolay ezberlenir: yalnız deseni (azalan çift sayılar), başlangıcı (50) ve sonu (14) hatırlamak yeter. Çok uzun dizileri hızla ve kolayca ezberleyebilseydik ikinci dizideki desen umurumuzda olmazdı; her sayıyı ezberlerdik. Uzun dizileri ezberlemenin zor olması desen tanımayı yararlı kılar. Otomatik kodlayıcıyı eğitimde kısıtlamanın onu verideki desenleri bulmaya itmesinin nedeni budur.

Bellek, algı ve örüntü eşleme arasındaki ilişkiyi 1970'lerin başında William Chase ve Herbert Simon incelemişti. Usta satranç oyuncularının, tahtaya yalnız beş saniye bakarak tüm taşların konumunu ezberleyebildiğini gözlemlediler; çoğu insan için imkânsız bir iştir bu. Ama bu yalnız taşlar gerçek oyunlardaki gibi gerçekçi konumlardayken geçerliydi; taşlar rastgele yerleştirildiğinde ustalar da ezberleyemedi. Satranç ustalarının belleği bizimkinden çok daha iyi değildir; oyundaki deneyimleri sayesinde satranç desenlerini daha kolay görürler. Desenleri fark etmek bilgiyi verimli biçimde saklamalarını sağlar.

### Kodlayıcı ve kod çözücü {#yapi}

Bu deneydeki satranç oyuncuları gibi bir otomatik kodlayıcı girdilere bakar, onları verimli bir gizil temsile çevirir ve sonra (umarız) girdilere çok yakın görünen bir şey üretir. Her otomatik kodlayıcı iki parçadan oluşur:
- girdileri gizil bir temsile çeviren **kodlayıcı** (tanıma ağı);
- iç temsili çıktılara çeviren **kod çözücü** (üretici ağ).

Otomatik kodlayıcı tipik olarak bir çok katmanlı algılayıcıyla (MLP) aynı mimariye sahiptir; tek fark, çıktı katmanındaki nöron sayısının girdi sayısına eşit olmasıdır. Slayttaki örnekte iki nöronlu tek bir gizli katman (kodlayıcı) ve üç nöronlu bir çıktı katmanı (kod çözücü) vardır. Çıktılara **yeniden yapılandırma** (reconstruction) denir, çünkü otomatik kodlayıcı girdileri yeniden kurmaya çalışır. Maliyet fonksiyonu, yeniden yapılandırmalar girdilerden farklı olduğunda modeli cezalandıran bir **yeniden yapılandırma kaybı** içerir.

İç temsil girdi verisinden daha düşük boyutlu olduğu için (3-B yerine 2-B) otomatik kodlayıcıya **eksik tamamlanmış** (undercomplete) denir. Eksik tamamlanmış bir otomatik kodlayıcı girdilerini koda doğrudan kopyalayamaz; yine de girdilerinin bir kopyasını üretmenin yolunu bulmak zorundadır. Girdi verisindeki en önemli öznitelikleri öğrenmeye (ve önemsizleri atmaya) zorlanır.

:::türetim Doğrusal otomatik kodlayıcının optimumu
Kodlayıcı ve kod çözücü doğrusal, kayıp kare hata olsun. M5.1'de gösterildiği gibi en iyi çözüm, veri kovaryansının ilk $k$ özvektörünün gerdiği alt uzaya izdüşümdür (Eckart–Young) ve hata atılan özdeğerlerin toplamıdır. El hesabı: kovaryansın özdeğerleri 5, 3, 1, 0.5 ve 2 boyutlu kod için en küçük beklenen hata $1+0.5=\mathbf{1.5}$'tir. Doğrusal olmayan aktivasyonlar ve daha derin ağlar eğri yüzeyler (manifold) öğrenebilir: PCA'nın doğrusal olmayan genellemesidir.
:::

**MNIST ile iki boyutlu kod.** Slayt 17–18'deki örnekte tam bağlı bir otomatik kodlayıcı MNIST'te iki boyutlu bir gizil uzayla eğitilir; kodlar çizilince aynı rakamlar kabaca bir arada kümelenir, ama gizil uzay düzensizdir (M5.1'in giriş bölümü). Görüntü olduğu için şekil metne dökülemedi.

### Yığılmış otomatik kodlayıcılar ve ağırlık bağlama {#yigilmis}

Diğer sinir ağları gibi otomatik kodlayıcıların da birden fazla gizli katmanı olabilir. Bu durumda **yığılmış** (ya da derin) otomatik kodlayıcı denir. Daha çok katman eklemek daha karmaşık kodlar öğrenmeye yardım eder.

Ama otomatik kodlayıcıyı çok güçlü yapmamaya dikkat etmek gerekir. Her girdiyi tek bir keyfi sayıya eşleyecek (kod çözücü de ters eşlemeyi öğrenecek) kadar güçlü bir kodlayıcı düşünün. Böyle bir otomatik kodlayıcı eğitim verisini kusursuz yeniden kurar, ama bu süreçte yararlı hiçbir temsil öğrenmez ve yeni örneklere iyi genellemesi de olası değildir.

Yığılmış bir otomatik kodlayıcının mimarisi tipik olarak merkezdeki gizli katmana (kod katmanı) göre **simetriktir**. Örneğin MNIST için: 784 girdi, 100 nöronlu bir gizli katman, 30 nöronlu merkezi gizli katman, yine 100 nöronlu bir gizli katman ve 784 nöronlu bir çıktı katmanı.

<figure class="fig wide"><svg viewBox="0 0 640 284" role="img" aria-label="Simetrik yığılmış otomatik kodlayıcı: 784, 100, 30, 100, 784 nöronlu katmanlar; kod çözücü ağırlıkları kodlayıcı ağırlıklarının devriği olarak bağlanır"><rect x="40" y="15" width="44" height="210" rx="5" class="soft-a"/><text x="62" y="9" text-anchor="middle" class="sm">784</text><text x="62" y="232" text-anchor="middle" class="sm">girdi</text><rect x="160" y="94" width="44" height="53" rx="5" class="soft-a"/><text x="182" y="88" text-anchor="middle" class="sm">100</text><text x="182" y="232" text-anchor="middle" class="sm">gizli 1</text><rect x="280" y="102" width="44" height="37" rx="5" class="soft-p"/><text x="302" y="96" text-anchor="middle" class="sm">30</text><text x="302" y="232" text-anchor="middle" class="sm">kod</text><rect x="400" y="94" width="44" height="53" rx="5" class="soft-a"/><text x="422" y="88" text-anchor="middle" class="sm">100</text><text x="422" y="232" text-anchor="middle" class="sm">gizli 3</text><rect x="520" y="15" width="44" height="210" rx="5" class="soft-a"/><text x="542" y="9" text-anchor="middle" class="sm">784</text><text x="542" y="232" text-anchor="middle" class="sm">çıktı</text><text x="122" y="40" text-anchor="middle" class="ta">W₁</text><text x="242" y="40" text-anchor="middle" class="ta">W₂</text><text x="362" y="40" text-anchor="middle" class="tb">W₃ = W₂ᵀ</text><text x="482" y="40" text-anchor="middle" class="tb">W₄ = W₁ᵀ</text><text x="40" y="256" class="sm">Kodlayıcı (mavi) 784 → 100 → 30, kod çözücü aynalanır. Bağlama (turuncu): W<tspan baseline-shift="-3" font-size="75%">N−L+1</tspan> = W<tspan baseline-shift="-3" font-size="75%">L</tspan>ᵀ; bias’lar ayrı.</text><text x="40" y="274" class="sm">Parametre: bağsız 163 814, bağlı 82 414 (ağırlıklar yarıya iner).</text></svg><figcaption>Şekil 1 — MNIST için simetrik yığılmış otomatik kodlayıcı (slayt 21) ve ağırlık bağlama (slayt 27–28). Kod katmanı (yeşil) girdiden çok küçüktür: eksik tamamlanmış (undercomplete). Bağlı modelde kod çözücünün her katmanı, karşılık gelen kodlayıcı katmanının devriğini kullanır.</figcaption></figure>

**Yeniden yapılandırmaları görselleştirmek.** Bir otomatik kodlayıcının doğru eğitildiğinden emin olmanın yollarından biri girdileri ve çıktıları karşılaştırmaktır; farklar çok büyük olmamalıdır. Slayt 22–23'teki örnekte yeniden yapılandırmalar tanınabilir ama biraz fazla kayıplıdır. Modeli daha uzun eğitmek, kodlayıcıyı ve kod çözücüyü derinleştirmek ya da kodları büyütmek gerekebilir. Ama ağı çok güçlü yaparsak verideki hiçbir yararlı deseni öğrenmeden kusursuz yeniden yapılandırmalar üretmeyi başarabilir.

**Ağırlık bağlama.** Otomatik kodlayıcı az önceki gibi düzgün biçimde simetrikse yaygın bir teknik, kod çözücü katmanlarının ağırlıklarını kodlayıcı katmanlarının ağırlıklarına **bağlamaktır**. Bu, modeldeki ağırlık sayısını yarıya indirir, eğitimi hızlandırır ve aşırı uyum riskini sınırlar. Somut olarak, otomatik kodlayıcının (girdi katmanı sayılmadan) toplam $N$ katmanı varsa ve $\mathbf W_L$ $L$. katmanın bağlantı ağırlıklarıysa (katman 1 ilk gizli katman, katman $N/2$ kod katmanı, katman $N$ çıktı katmanı), kod çözücü katmanlarının ağırlıkları basitçe şöyle tanımlanır:
$$\mathbf W_{N-L+1}=\mathbf W_L^\top,\qquad L=1,2,\dots,N/2 .$$

:::örnek El hesabı (modülün el hesabı alanları)
784 → 100 → 30 → 100 → 784, biaslar dahil. Bağsız: $(784\cdot100+100)+(100\cdot30+30)+(30\cdot100+100)+(100\cdot784+784)=78\,500+3\,030+3\,100+79\,184=\mathbf{163\,814}$. Bağlı: kod çözücünün ağırlıkları paylaşılır, yalnız biasları ayrıdır: $78\,500+3\,030+100+784=\mathbf{82\,414}$. Parametre sayısı neredeyse yarıya iner ($163\,814/82\,414=1.99$).
:::

:::sezgi Bağlı ağırlıklar neden mantıklı?
Doğrusal durumda en iyi kod çözücü, kodlayıcının **sözde tersidir**; kodlayıcının satırları ortonormalse bu tam olarak devriktir ($\mathbf W^+=\mathbf W^\top$). PCA'da izdüşüm $\mathbf U_k^\top$, geri dönüş $\mathbf U_k$'dır. Ağırlık bağlama, modeli bu yapıya yakın tutan bir düzenlileştirmedir. Slayt 28, bağlı modelin neredeyse yarı parametreyle bağsız modelden çok az daha düşük yeniden yapılandırma hatası verebildiğini söyler. Bu bir gözlemdir, genel bir kural değildir: bağlama kapasiteyi azaltır; az veride genellemeye yardım edebilir, bol veride bağsız model biraz daha iyi kurabilir. Aşama d iki modeli aynı veride karşılaştırıyor.
:::

### Açgözlü katman katman eğitim {#acgozlu}

Tüm yığılmış otomatik kodlayıcıyı tek seferde eğitmek yerine her seferinde bir sığ otomatik kodlayıcı eğitmek, sonra hepsini tek bir yığılmış otomatik kodlayıcıda üst üste koymak da mümkündür (adı da buradan gelir).
1. Eğitimin ilk aşamasında ilk otomatik kodlayıcı girdileri yeniden kurmayı öğrenir.
2. Tüm eğitim kümesi bu ilk otomatik kodlayıcıyla kodlanır; bu, yeni (sıkıştırılmış) bir eğitim kümesi verir. İkinci otomatik kodlayıcı bu yeni veri kümesinde eğitilir.
3. Son olarak tüm bu otomatik kodlayıcılardan büyük bir sandviç kurulur: önce her otomatik kodlayıcının gizli katmanları, sonra çıktı katmanları ters sırada yığılır. Böylece son yığılmış otomatik kodlayıcı elde edilir. Bu yolla daha fazla otomatik kodlayıcı eğitip çok derin bir yığılmış otomatik kodlayıcı kurulabilir.

<figure class="fig wide"><svg viewBox="0 0 640 222" role="img" aria-label="Açgözlü katman katman eğitim: önce girdiyi kuran sığ otomatik kodlayıcı, sonra onun kodlarını kuran ikinci sığ otomatik kodlayıcı, sonunda ikisinin sandviç biçiminde yığılması"><text x="20" y="20" class="sm">1. aşama: sığ AE, girdiyi kurar</text><rect x="20" y="36" width="150" height="24" rx="4" class="soft-a"/><text x="95" y="52" text-anchor="middle" class="sm">çıktı 784</text><rect x="20" y="66" width="150" height="24" rx="4" class="soft-p"/><text x="95" y="82" text-anchor="middle" class="sm">gizli 100</text><rect x="20" y="96" width="150" height="24" rx="4" class="soft-a"/><text x="95" y="112" text-anchor="middle" class="sm">girdi 784</text><text x="225" y="20" class="sm">2. aşama: kodları kuran sığ AE</text><rect x="225" y="36" width="150" height="24" rx="4" class="soft-a"/><text x="300" y="52" text-anchor="middle" class="sm">çıktı 100</text><rect x="225" y="66" width="150" height="24" rx="4" class="soft-p"/><text x="300" y="82" text-anchor="middle" class="sm">gizli 30</text><rect x="225" y="96" width="150" height="24" rx="4" class="soft-b"/><text x="300" y="112" text-anchor="middle" class="sm">girdi: 100 boyutlu kodlar</text><text x="430" y="20" class="sm">3. aşama: sandviç (yığılmış AE)</text><rect x="430" y="36" width="150" height="24" rx="4" class="soft-a"/><text x="505" y="52" text-anchor="middle" class="sm">çıktı 784 (1. AE)</text><rect x="430" y="66" width="150" height="24" rx="4" class="soft-a"/><text x="505" y="82" text-anchor="middle" class="sm">gizli 100 (2. AE çıktısı)</text><rect x="430" y="96" width="150" height="24" rx="4" class="soft-p"/><text x="505" y="112" text-anchor="middle" class="sm">kod 30 (2. AE)</text><rect x="430" y="126" width="150" height="24" rx="4" class="soft-p"/><text x="505" y="142" text-anchor="middle" class="sm">gizli 100 (1. AE)</text><rect x="430" y="156" width="150" height="24" rx="4" class="soft-a"/><text x="505" y="172" text-anchor="middle" class="sm">girdi 784</text><text x="20" y="210" class="sm">Her aşamada yalnız sığ bir AE eğitilir; bir öncekinin kodları bir sonrakinin eğitim verisidir.</text></svg><figcaption>Şekil 2 — Açgözlü katman katman eğitim (slayt 29–31). Önce gizli katmanlar sırayla, sonra çıktı katmanları ters sırayla yığılır. BN, ReLU ve artık bağlantılardan önce derin ağları eğitmenin başlıca yoluydu (Hinton ve Salakhutdinov 2006; Bengio ve ark. 2007).</figcaption></figure>

Bu teknik bugün pek kullanılmaz, ama "açgözlü katman katman eğitim" (greedy layer-wise training) diyen makalelerle karşılaşabilirsiniz; ne anlama geldiğini bilmek iyidir. Tarihsel önemi büyüktür. 2006'da Hinton ve Salakhutdinov ile Bengio ve arkadaşları, derin ağların katman katman denetimsiz ön eğitimle başlatılınca eğitilebildiğini gösterdi. Bu, "derin öğrenme" döneminin başlangıçlarından biri sayılır. ReLU, iyi başlatma (M3.5), normalizasyon (M3.6) ve artık bağlantılar (M4.2) geldikten sonra derin ağlar doğrudan eğitilebilir hâle geldi ve bu ön eğitim gereksizleşti.

### Otomatik kodlayıcıyla denetimsiz ön eğitim {#ontegitim}

Karmaşık bir denetimli görevle uğraşıyorsunuz ama çok fazla etiketli eğitim veriniz yok. Bir çözüm, benzer bir görevi yapan bir sinir ağı bulup alt katmanlarını yeniden kullanmaktır (M3.9). Bu, az eğitim verisiyle yüksek başarımlı bir model eğitmeyi mümkün kılar, çünkü ağınız tüm alt düzey öznitelikleri öğrenmek zorunda kalmaz; mevcut ağın öğrendiği öznitelik dedektörlerini yeniden kullanır.

Benzer biçimde, büyük bir veri kümeniz varsa ama çoğu etiketsizse, önce tüm veriyi kullanarak yığılmış bir otomatik kodlayıcı eğitebilirsiniz. Sonra asıl göreviniz için bir sinir ağı kurarken alt katmanları yeniden kullanır ve etiketli veriyle eğitirsiniz. Bu adım için etiket gerekmez. Sınıflandırıcıyı eğitirken gerçekten çok az etiketli veriniz varsa önceden eğitilmiş katmanları (en azından alt olanları) dondurmak isteyebilirsiniz. Aşama d bunu sklearn rakamlarında yalnız 50 etiketle deniyor: sıfırdan eğitim, ön eğitim + ince ayar ve donuk kodlayıcı (AE ve DAE) karşılaştırılıyor. Kazancın büyüklüğü etiket sayısına ve rastgeleliğe bağlıdır; etiket çoğaldıkça fark kapanır. Bu fikrin modern hâli M4.16'daki dil modeli ön eğitimidir.

### Konvolüsyonel ve rekürent otomatik kodlayıcılar {#konv}

Görüntülerle çalışıyorsanız şimdiye kadar gördüğümüz otomatik kodlayıcılar (görüntüler çok küçük değilse) iyi çalışmaz. Daha önce gördüğümüz gibi konvolüsyonel sinir ağları görüntülerle çalışmaya yoğun ağlardan çok daha uygundur. Görüntüler için bir otomatik kodlayıcı kurmak istiyorsanız (ör. denetimsiz ön eğitim ya da boyut indirgeme için) bir **konvolüsyonel otomatik kodlayıcı** kurmanız gerekir. Kodlayıcı, konvolüsyon ve havuzlama katmanlarından oluşan sıradan bir CNN'dir. Tipik olarak girdilerin uzamsal boyutunu (yükseklik ve genişlik) azaltırken derinliği (öznitelik haritası sayısı) artırır. Kod çözücü tersini yapmalıdır: görüntüyü büyütmeli ve derinliği orijinal boyutlara indirmelidir. Bunun için transposed konvolüsyon katmanları ya da yukarı örnekleme ile konvolüsyonun birleşimi kullanılabilir (M4.10; dama tahtası etkisi). El hesabı: 7 × 7'lik bir harita, çekirdeği 3, adımı 2, dolgusu 1 ve `output_padding`'i 1 olan bir transposed konvolüsyonla $(7-1)\cdot2-2\cdot1+3+1=\mathbf{14}$ kenarlı olur.

Zaman serileri ya da metin gibi diziler için (ör. denetimsiz öğrenme ya da boyut indirgeme amacıyla) bir otomatik kodlayıcı kurmak istiyorsanız rekürent sinir ağları yoğun ağlardan daha uygun olabilir. **Rekürent otomatik kodlayıcı** kurmak basittir: kodlayıcı tipik olarak girdi dizisini tek bir vektöre sıkıştıran bir dizi→vektör RNN'dir, kod çözücü ise tersini yapan bir vektör→dizi RNN'dir (M4.13'teki kodlayıcı–kod çözücünün kendini çeviren hâli).

### Aşırı tamamlanmış kodlar: gürültü giderme ve seyreklik {#asiri}

Şimdiye kadar çeşitli otomatik kodlayıcı türleri (temel, yığılmış, konvolüsyonel, rekürent) ve onları eğitmenin yollarını (tek seferde ya da katman katman) gördük. Otomatik kodlayıcıyı ilginç öznitelikler öğrenmeye zorlamak için kod katmanının boyutunu sınırladık, onu eksik tamamlanmış yaptık. Aslında kullanılabilecek başka birçok kısıt türü vardır. Bunların bazıları kod katmanının girdiler kadar büyük, hatta daha büyük olmasına izin verir; ortaya **aşırı tamamlanmış** (overcomplete) bir otomatik kodlayıcı çıkar.

**Gürültü giderici otomatik kodlayıcılar.** Otomatik kodlayıcıyı yararlı öznitelikler öğrenmeye zorlamanın bir başka yolu girdilerine gürültü eklemek ve onu orijinal, gürültüsüz girdileri kurmaya eğitmektir. Bu fikir 1980'lerden beri vardır (ör. Yann LeCun'un 1987 yüksek lisans tezinde geçer). Pascal Vincent ve arkadaşları 2008'deki bir makalede otomatik kodlayıcıların öznitelik çıkarmak için de kullanılabileceğini gösterdi; 2010'daki bir makalede yığılmış gürültü giderici otomatik kodlayıcıları tanıttılar. Gürültü, girdilere eklenmiş saf Gauss gürültüsü olabilir; ya da dropout'taki gibi rastgele kapatılmış girdiler (maskeleme). Slayt 38'deki örnekte piksellerin yarısı kapatılmış gürültülü görüntüler ve dropout tabanlı gürültü giderici otomatik kodlayıcının yeniden kurduğu görüntüler var. Otomatik kodlayıcının, girdide aslında bulunmayan ayrıntıları (ör. beyaz gömleğin üst kısmı) nasıl tahmin ettiğine dikkat edin. (Görüntü metne dökülemedi.)

:::türetim Gürültü giderici neyi öğrenir? Tweedie ve skor
$\tilde{\mathbf x}=\mathbf x+\sigma\boldsymbol\varepsilon$ bozmasıyla kare hata en küçüklenirse, sonsuz kapasiteli bir gürültü gidericinin optimumu koşullu ortalamadır: $r^*(\tilde{\mathbf x})=\mathbb E[\mathbf x\mid\tilde{\mathbf x}]$ (M0.4: kare hatayı en küçükleyen tahmin koşullu beklentidir). Gürültülü verinin yoğunluğu $p_\sigma(\tilde{\mathbf x})=\int p(\mathbf x)\,\mathcal N(\tilde{\mathbf x};\mathbf x,\sigma^2\mathbf I)\,d\mathbf x$. Türevini alalım:
$$\nabla_{\tilde{\mathbf x}}p_\sigma(\tilde{\mathbf x})=\int p(\mathbf x)\,\mathcal N(\tilde{\mathbf x};\mathbf x,\sigma^2\mathbf I)\,\frac{\mathbf x-\tilde{\mathbf x}}{\sigma^2}\,d\mathbf x=p_\sigma(\tilde{\mathbf x})\,\frac{\mathbb E[\mathbf x\mid\tilde{\mathbf x}]-\tilde{\mathbf x}}{\sigma^2}.$$
Dolayısıyla
$$r^*(\tilde{\mathbf x})-\tilde{\mathbf x}=\sigma^2\,\nabla_{\tilde{\mathbf x}}\log p_\sigma(\tilde{\mathbf x})\qquad(\text{Tweedie formülü}).$$
En iyi gürültü gidericinin artığı, gürültülü veri yoğunluğunun **skoruyla** orantılıdır: noktayı verinin daha olası olduğu yöne iter (Alain ve Bengio 2014; Vincent 2011). Difüzyon modellerinin gürültü tahmini (M5.3) tam olarak budur. Bir difüzyon modeli, her gürültü düzeyinde ayrı bir gürültü giderici otomatik kodlayıcıdır.
:::

Aşağıdaki deneyde iki tepeli bir 1-B veri için en iyi gürültü gidericiyi kesin hesapla. Artığın skora eşit olduğunu ve belirsiz bölgelerde (iki tepenin ortası) çıktının neden ortalama (bulanık) olduğunu gör.

::lab l1

**Seyrek otomatik kodlayıcılar.** İyi öznitelik çıkarımına sıklıkla yol açan başka bir kısıt **seyrekliktir**. Maliyet fonksiyonuna uygun bir terim eklenerek otomatik kodlayıcı kod katmanındaki etkin nöron sayısını azaltmaya itilir. Örneğin kod katmanında ortalama yalnız %5 belirgin biçimde etkin nöron bulundurmaya zorlanabilir. Bu, otomatik kodlayıcıyı her girdiyi az sayıda aktivasyonun birleşimi olarak temsil etmeye zorlar. Sonuç olarak kod katmanındaki her nöron tipik olarak yararlı bir özniteliği temsil eder. Slaytın benzetmesiyle: ayda yalnız birkaç kelime konuşabilseydiniz, muhtemelen dinlenmeye değer olanları seçmeye çalışırdınız.

Basit bir yaklaşım şudur: kod katmanında sigmoid aktivasyon kullan (kodları 0 ile 1 arasında sınırlamak için), büyük bir kod katmanı kullan (ör. 300 birim) ve kod katmanının aktivasyonlarına biraz **L1** düzenlileştirmesi ekle. Kod çözücü sıradan bir kod çözücüdür. L2 yerine L1 normu kullanmak, ağı tüm kodları küçültmek yerine en önemli kodları korumaya ve girdi görüntüsü için gerekmeyenleri sıfırlamaya iter.

:::türetim L1 neden sıfırlar, L2 neden yalnız küçültür?
Tek bir kod bileşeni için ham değer $u$ ile ceza arasındaki dengeyi düşünelim.
- $\min_z\frac12(z-u)^2+\lambda z^2$'nin çözümü $z=u/(1+2\lambda)$'dır. Her bileşen aynı oranda küçülür, hiçbiri tam sıfır olmaz.
- $\min_z\frac12(z-u)^2+\lambda|z|$'nin çözümü **yumuşak eşiklemedir**: $z=\mathrm{sign}(u)\max(|u|-\lambda,0)$. $|u|\le\lambda$ olan bileşenler tam sıfır olur; büyükler $\lambda$ kadar küçülür.

Sebep geometriktir: $|z|$'nin 0'da bir köşesi vardır ve alt gradyanı $[-\lambda,\lambda]$ aralığının tamamıdır. Küçük bir $u$ bu aralığa sığar ve optimum 0'da kalır (M1.3'teki lasso ile aynı mekanizma). Yinelemeli çözüm (ISTA/FISTA), her adımda bir gradyan adımı ve ardından bu yumuşak eşiklemeyi uygular.
:::

**KL seyreklik cezası.** Başka bir yaklaşım her kod nöronunun eğitim batch'i üzerindeki ortalama aktivasyonunu $\hat\rho_j$ ölçmek ve hedef seyreklik $\rho$'dan (ör. 0.05) uzaklaşmasını KL ıraksamasıyla cezalandırmaktır: $\beta\sum_j\mathrm{KL}(\rho\,\|\,\hat\rho_j)$. Burada $\mathrm{KL}(\rho\|\hat\rho)=\rho\log\frac\rho{\hat\rho}+(1-\rho)\log\frac{1-\rho}{1-\hat\rho}$, iki Bernoulli dağılımı arasındaki KL'dir. El hesabı: $\rho=0.05$, $\hat\rho=0.2$ için $\mathrm{KL}=0.05\log0.25+0.95\log1.1875=\mathbf{0.0939}$ ve $\partial\mathrm{KL}/\partial\hat\rho=-\rho/\hat\rho+(1-\rho)/(1-\hat\rho)=-0.25+1.1875=\mathbf{0.9375}$. Pozitif gradyan aktivasyonu düşürmeye iter. L1'den farkı şudur: KL, tek tek örneklerin değil **ortalama** aktivasyonun hedefe uymasını ister ve aktivasyon 0'a yaklaştıkça ceza sınırsız büyür. Aşama c bu cezayı bağlı ağırlıklı bir otomatik kodlayıcıda elle türetilmiş gradyanlarla eğitiyor.

Aşağıdaki deneyde aşırı tamamlanmış bir sözlükle (64 örnekli sinyal, 128 atom) kod bul. L1 cezasının birkaç atomla açıklayan seyrek bir kod, L2 cezasının ise neredeyse tüm atomlara yayılan yoğun bir kod verdiğini gör.

::lab l2

### Ekler ve uygulamalar {#ekler}

- **Doğrusal gürültü giderici otomatik kodlayıcı** (ek x1): $\hat{\mathbf x}=\mathbf W(\mathbf x+\boldsymbol\varepsilon)$'nin kapalı form çözümü $\mathbf W^*=\boldsymbol\Sigma(\boldsymbol\Sigma+\sigma^2\mathbf I)^{-1}$'dir. Özyönlerde bir küçültme yapar: özdeğeri $\lambda$ olan yön $\lambda/(\lambda+\sigma^2)$ ile çarpılır. Gürültü, ridge regresyonla aynı düzenlileştirmeyi yapar (M1.3).
- **Büzüşen (contractive) otomatik kodlayıcı** (ek x2; Rifai ve ark. 2011): kodlayıcının Jacobian'ının Frobenius normunu cezalandırır, $\|\partial\mathbf h/\partial\mathbf x\|_F^2$. Kodlar girdideki küçük değişimlere duyarsız olur. Gürültü giderici otomatik kodlayıcı, küçük gürültüde buna benzer bir düzenlileştirme yapar.
- **Anomali tespiti** (ek x3): yalnız normal verilerle eğitilmiş bir otomatik kodlayıcı (ya da PCA), alışılmadık örnekleri kötü yeniden kurar. Yeniden yapılandırma hatası bir anomali skorudur; başarım ROC eğrisi altındaki alanla (AUC) ölçülür.

### PyTorch'ta {#pytorch}

| kavram | PyTorch | not |
|---|---|---|
| yığılmış AE | `nn.Sequential(Linear(784, 100), ReLU(), Linear(100, 30))` + aynalanmış kod çözücü | |
| bağlı ağırlık | `F.linear(h, self.enc.weight.t(), self.dec_bias)` | kod çözücü bias'ı ayrı `nn.Parameter` |
| maskeleme gürültüsü | `x_noisy = x * (torch.rand_like(x) > p)` | dropout'un girdideki karşılığı |
| Gauss gürültüsü | `x_noisy = x + sigma * torch.randn_like(x)` | |
| L1 seyreklik | `loss = mse + lam * z.abs().mean()` | |
| KL seyreklik | `rho_hat = z.mean(0); kl = (rho * (rho / rho_hat).log() + (1 - rho) * ((1 - rho) / (1 - rho_hat)).log()).sum()` | sigmoid kod |
| konvolüsyonel AE | `Conv2d(..., stride=2)` / `ConvTranspose2d(..., stride=2, output_padding=1)` | M4.10 |
| kodlayıcıyı dondurma | `for p in ae.enc.parameters(): p.requires_grad_(False)` | az etiket |

### Özet {#ozet}

- Otomatik kodlayıcı = kodlayıcı + kod çözücü; özdeşliği bir kısıt altında öğrenir; kodlar bunun yan ürünüdür. Kısıt yoksa ya da ağ çok güçlüyse kopyalar ve yararlı bir şey öğrenmez.
- Eksik tamamlanmış doğrusal AE = PCA alt uzayı (el örneği: hata 1.5). Yığılmış AE simetriktir (784-100-30-100-784).
- Ağırlık bağlama $\mathbf W_{N-L+1}=\mathbf W_L^\top$: 163 814 → 82 414 parametre; doğrusal durumda devrik, ortonormal kodlayıcının tersidir.
- Açgözlü katman katman eğitim: sığ AE'leri sırayla eğit, sandviç biçiminde yığ. Tarihsel olarak derin öğrenmenin başlangıcı; bugün gereksiz.
- Denetimsiz ön eğitim: etiketsiz veriyle AE, alt katmanları az etiketli sınıflandırıcıda kullan (gerekirse dondur).
- Konvolüsyonel AE (kodlayıcı CNN, kod çözücü transposed konvolüsyon), rekürent AE (dizi→vektör→dizi).
- Gürültü giderici AE: en iyisi $\mathbb E[\mathbf x\mid\tilde{\mathbf x}]$; artığı $\sigma^2\nabla\log p_\sigma$ (Tweedie): difüzyonla aynı nesne.
- Seyrek AE: L1 yumuşak eşikleme ile sıfırlar, L2 yalnız küçültür; KL cezası ortalama aktivasyonu hedefe çeker (el örneği: 0.0939, gradyan 0.9375).
