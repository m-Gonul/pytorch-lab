### Yanlılık, varyans ve modern ağlar {#yanlilik}

M1.3'teki ayrışım her tahminci için geçerlidir: beklenen test hatası = yanlılık² + varyans + indirgenemez gürültü. Ders slaytlarının özeti: **modern yapay sinir ağları düşük yanlılıklı modellerdir; performans sorunları çoğunlukla varyansla ilgilidir.** Yeterince büyük bir ağ eğitim verisini neredeyse her zaman ezberleyebilir (M3.8, evrensel yaklaşım); soru ezberin genellenip genellenmediğidir. M3.8'deki çift iniş bu tablonun inceliğidir: parametre sayısı veri sayısını çok aştığında, uygun örtük ya da açık düzenlileştirmeyle varyans yeniden düşebilir.

Pratikte iki sayıya bakarak teşhis koyarsın: **eğitim hatası** ve **doğrulama hatası** (ve mümkünse ulaşılabilir hatanın bir tahmini — insan performansı, gürültü tabanı).

| belirti | teşhis | ilk denenecekler |
|---|---|---|
| eğitim hatası yüksek, doğrulama ona yakın | **yüksek yanlılık** (eksik uyum) | daha büyük/derin model, daha uzun eğitim, daha iyi optimizer/öğrenme oranı, daha az düzenlileştirme |
| eğitim hatası düşük, doğrulama çok daha yüksek | **yüksek varyans** (aşırı uyum) | daha çok veri (veya veri artırma), düzenlileştirme (M3.7), early stopping, daha küçük model |
| ikisi de yüksek ve aralarında büyük fark | ikisi birden | önce yanlılığı çöz, sonra varyansı |
| ikisi de düşük ve yakın | iyi | test kümesinde bir kez doğrula |

**Öğrenme eğrisi** (hata vs eğitim örneği sayısı) "daha çok veri yardım eder mi?" sorusunu yanıtlar: yüksek yanlılıkta iki eğri erkenden birbirine yaklaşır ve **yüksek bir düzeyde** düzleşir — veri eklemek boşunadır. Yüksek varyansta aralarında büyük bir boşluk vardır ve veri eklendikçe kapanır. **Kapasite eğrisi** (hata vs model büyüklüğü) aynı veride hangi büyüklüğün en iyi olduğunu gösterir.

::lab l1

:::ayrıntı Eğitim eğrisi neden yarım epoch kaymış görünür?
Keras ve çoğu eğitim döngüsü **eğitim hatasını epoch boyunca kayan ortalama** olarak, **doğrulama hatasını epoch sonunda** hesaplar. Epoch'un ilk batch'lerindeki (henüz daha kötü olan) ağın hataları eğitim ortalamasına girer; ağ epoch boyunca iyileştiği için eğitim eğrisi ortalamada yarım epoch **geriden** gelir. İki eğriyi karşılaştırırken eğitim eğrisini yarım epoch sola kaydır; yoksa ilk epoch'larda doğrulama hatası eğitim hatasından düşük görünebilir ve bu bir sızıntı belirtisi değildir. (Dropout gibi yalnız eğitimde açık katmanlar da eğitim hatasını yapay olarak yükseltir.)
:::

### Hiperparametreler {#hiperparametre}

Sinir ağlarının esnekliği aynı zamanda en büyük dezavantajlarından biridir: ayarlanacak çok hiperparametre vardır. Ders slaytlarındaki listeyi, bugünkü pratikle birlikte:

| hiperparametre | durum | pratik başlangıç |
|---|---|---|
| ağ mimarisi | probleme özgü | literatürdeki en iyi (SOTA) modellere bak, uygun olanı seç |
| katman sayısı | uzlaşı yok | genişliği artırmak yerine **derinliği** artırmak genelde daha çok getiri sağlar (M3.8) |
| katman başına nöron | uzlaşı yok | bir "esneme pantolonu": biraz büyük seç, early stopping ve düzenlileştirmeyle dizginle |
| aktivasyon | yerleşik | ReLU türevlerinden biri (M3.5) |
| öğrenme oranı | **en önemlisi** | büyük başla, azalt; kosinüs takvimi bugün iyi bir varsayılan (M3.4) |
| ağırlık başlatma | aktivasyona bağlı | ReLU → He, tanh → Xavier/Glorot (M3.5) |
| batch boyu | uzlaşı yok, donanıma bağlı | belleğin sığdırdığı büyüklük; öğrenme oranını birlikte ayarla |
| normalizasyon | yerleşik | BatchNorm; ya da SELU ile öz-normalizasyon (M3.5–M3.6) |

En iyi kombinasyonu nasıl bulursun? Kısa cevap: **deneme yanılma** — hiperparametre ayarı hâlâ aktif bir araştırma alanıdır. Ama deneme yanılmanın iyi ve kötü yolları vardır:

- **Önce öğrenme oranı.** Diğer her şey sabitken en çok fark yaratan odur. **Öğrenme oranı aralık testi** (Smith 2017): η'yı birkaç yüz adım boyunca $10^{-6}$'dan 10'a üstel olarak artır, kaybı çiz; kaybın en hızlı düştüğü bölgenin biraz altını (tipik olarak patlamanın başladığı η'nın ≈ 1/10'u) seç.
- **Izgara değil rastgele arama.** Bergstra ve Bengio (2012): hiperparametrelerin genellikle yalnız birkaçı gerçekten önemlidir; 9 noktalı bir 3×3 ızgara önemli eksende yalnız 3 farklı değer dener, 9 rastgele nokta 9 farklı değer dener. Aynı bütçeyle rastgele arama önemli boyutu çok daha iyi tarar.
- **Log ölçek.** Öğrenme oranı, weight decay, λ gibi çarpımsal etkili hiperparametreleri **log-uniform** örnekle ($10^{-5}$–$10^{-1}$ arası her on katlık dilime eşit pay).
- **Kabadan inceye.** Geniş aralıkta kısa denemeler, sonra iyi bölgede daha dar ve uzun denemeler. Kötü denemeleri erken kesen yöntemler (**successive halving**, **Hyperband**) bütçeyi umut vadeden yapılandırmalara aktarır; Optuna, Ray Tune gibi kütüphaneler bunları hazır sunar.
- **Bir seferde bir değişiklik** ve her denemenin kaydı (yapılandırma + tohum + sonuç); aynı yapılandırmayı birkaç tohumla çalıştırmadan küçük farklara güvenme.

### Doğrulama, test ve veri sızıntısı {#sizinti}

Herhangi bir istatistiksel problemde mevcut veri genellikle üç kümeye ayrılır:

- **Eğitim kümesi:** modeli eğitmek için (ör. geri yayılımla).
- **Doğrulama kümesi:** aşırı uyumu tespit etmek ve önlemek, hiperparametreleri ayarlamak, model seçmek için.
- **Test kümesi:** modeli **yalnız tüm eğitim bittikten sonra** değerlendirmek için.

Veri setleri çoğu zaman ayrı eğitim ve test kümeleriyle gelir; eğitim kümesi ayrıca bölünür (ör. %90 eğitim – %10 doğrulama). Az veride tek bir doğrulama kümesi gürültülüdür; **k-katlı çapraz doğrulama** geliştirme verisini $k$ kata böler, her katı sırayla doğrulama yapar ve skorların ortalamasını alır (M1.3).

<figure class="fig wide"><svg viewBox="0 0 600 242" role="img" aria-label="Veri, eğitim-doğrulama ve test olarak bölünür; geliştirme verisi beş kata ayrılır ve her kat sırayla doğrulama olur"><rect x="20" y="20" width="448" height="30" rx="4" class="soft-a ca"/><rect x="468" y="20" width="112" height="30" rx="4" class="soft-p cp"/><text x="244" y="40" text-anchor="middle">eğitim + doğrulama (geliştirme verisi)</text><text x="524" y="40" text-anchor="middle" class="tp">test</text><text x="408" y="68" class="sm">test kümesi kilitli: yalnız en sonda, bir kez</text><rect x="21.0" y="90" width="87.6" height="20" rx="3" class="soft-b cb"/><rect x="110.6" y="90" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="200.2" y="90" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="289.8" y="90" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="379.4" y="90" width="87.6" height="20" rx="3" class="soft-a ca"/><text x="476.0" y="104" class="sm">kat 1: doğrulama skoru s₁</text><rect x="21.0" y="116" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="110.6" y="116" width="87.6" height="20" rx="3" class="soft-b cb"/><rect x="200.2" y="116" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="289.8" y="116" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="379.4" y="116" width="87.6" height="20" rx="3" class="soft-a ca"/><text x="476.0" y="130" class="sm">kat 2: doğrulama skoru s₂</text><rect x="21.0" y="142" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="110.6" y="142" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="200.2" y="142" width="87.6" height="20" rx="3" class="soft-b cb"/><rect x="289.8" y="142" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="379.4" y="142" width="87.6" height="20" rx="3" class="soft-a ca"/><text x="476.0" y="156" class="sm">kat 3: doğrulama skoru s₃</text><rect x="21.0" y="168" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="110.6" y="168" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="200.2" y="168" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="289.8" y="168" width="87.6" height="20" rx="3" class="soft-b cb"/><rect x="379.4" y="168" width="87.6" height="20" rx="3" class="soft-a ca"/><text x="476.0" y="182" class="sm">kat 4: doğrulama skoru s₄</text><rect x="21.0" y="194" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="110.6" y="194" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="200.2" y="194" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="289.8" y="194" width="87.6" height="20" rx="3" class="soft-a ca"/><rect x="379.4" y="194" width="87.6" height="20" rx="3" class="soft-b cb"/><text x="476.0" y="208" class="sm">kat 5: doğrulama skoru s₅</text><text x="20" y="232" class="sm">her katta model sıfırdan eğitilir; CV skoru = (s₁ + … + s₅)/5; ön işleme ve öznitelik seçimi de her katın içinde</text></svg><figcaption>Şekil 1 — Üç küme ve 5-katlı çapraz doğrulama. Turuncu kat o turun doğrulama kümesidir. Hiperparametreler CV skoruna göre seçilir; seçilen yapılandırma tüm geliştirme verisiyle yeniden eğitilir ve test kümesinde <b>bir kez</b> değerlendirilir.</figcaption></figure>

**Hiperparametreleri test kümesinde ayarlayabilir miyiz?** Hayır. Test kümesine göre yapılan her seçim (model, epoch, eşik, öznitelik) test kümesini bir **doğrulama kümesine** çevirir: o kümede ölçülen performans artık iyimser yanlıdır ve gerçek, görülmemiş veri performansını tahmin etmez. Aynı test kümesine çok kez bakıp en iyisini seçmek, onu yavaş yavaş "eğitmektir" (Kaggle'da açık sıralama tablosuna aşırı uyum bunun bilinen hâlidir). Hem model seçimi hem dürüst bir performans tahmini gerekiyorsa **iç içe CV** (nested CV): iç döngü seçer, dış döngü değerlendirir.

:::tanım Veri sızıntısı (data leakage)
Hem eğitimde hem değerlendirmede **bir şekilde** bulunan herhangi bir veri kaynağı — ya da tahmin anında gerçekte bulunmayacak bir bilginin modele ulaşması. Sızıntı varsa modelin her değerlendirmesi **geçersizdir**.
:::

Sızıntı neredeyse hiçbir zaman açıkça "test verisini eğitime koymak" biçiminde olmaz; sinsidir:

| tür | örnek | çare |
|---|---|---|
| **ön işleme** | ölçekleme ortalaması/std'si, eksik değer doldurma, PCA, öznitelik seçimi **tüm** veride hesaplanıp sonra bölünür | her şeyi yalnız eğitim kısmında `fit` et (scikit-learn `Pipeline`; CV'de her katın içinde) |
| **grup** | aynı hastanın birden çok görüntüsü, aynı konuşmacının kayıtları, aynı belgenin kopyaları hem eğitimde hem testte | grup düzeyinde böl (`GroupKFold`); kopyaları (yakın kopyalar dahil) temizle |
| **zaman** | zaman serisinde rastgele bölme: model geleceği görüp geçmişi tahmin eder | ileri zincirleme (`TimeSeriesSplit`): eğitim hep testten önce |
| **hedef** | hedefin sonucu olan bir öznitelik (ör. "tedavi başlangıç tarihi" ile hastalık tahmini; iade edildikten sonra oluşan bir alan) | her özniteliğe "tahmin anında bu değer elimde olur mu?" diye sor |

Aşağıdaki deneyde iki sızıntıyı sayılarla gör. **Grup sızıntısı:** her "hastanın" birkaç çok benzer örneği var; rastgele bölmede 1-NN test örneğinin eğitimdeki kardeşini bulur ve etiketini kopyalar. **Öznitelik seçimi sızıntısı:** etiketler tamamen rastgele (gerçek doğruluk %50), ama öznitelikleri tüm veriye bakarak seçince CV %90'ların üstünde bir doğruluk "bulur" (Hastie, Tibshirani ve Friedman'ın *The Elements of Statistical Learning* kitabındaki §7.10.2 klasik örneği; x1 ekinde kodla yeniden üretiyorsun).

::lab l2

### Dengesiz sınıflar ve veri ağırlıklandırma {#agirlik}

Bazı sınıfların diğerlerinden çok daha fazla örneği varsa eğitim kümesine **çarpık** (skewed) denir: dolandırıcılık tespitinde %0.1 pozitif, tıbbi taramada %1 hasta. "Hep çoğunluk sınıfını söyle" modeli %99 doğruluk alır ve işe yaramaz — **doğruluk** bu durumda yanıltıcı bir ölçüdür; azınlık sınıfının **recall** ve **precision**'ına, dengeli doğruluğa, PR eğrisine bak (M1.4).

Çerçeveler eğitim sırasında sınıflara farklı ağırlık vermeyi sağlar:
- **Keras:** `fit()`'e `class_weight` — az temsil edilen sınıflara büyük, çok temsil edilenlere küçük ağırlık; kayıp hesaplanırken kullanılır. Örnek başına ağırlık için `sample_weight` (ikisi birden verilirse çarpılır). Örnek ağırlıkları, uzmanların etiketlediği örneklere kitle kaynaklı etiketlerden daha çok güvenmek gibi durumlarda da yararlıdır. Doğrulama kümesi için örnek ağırlığı (sınıf ağırlığı değil) `validation_data` demetinin üçüncü öğesi olarak verilebilir.
- **PyTorch:** `nn.CrossEntropyLoss(weight=w)` (sınıf ağırlıkları), örnek ağırlığı için `reduction="none"` ile kaybı al, ağırlıklarla çarpıp ortala; ya da kaybı değil **örneklemeyi** değiştir: `WeightedRandomSampler` azınlık örneklerini daha sık çeker.

:::türetim Sınıf ağırlığı olasılıkları nasıl değiştirir?
İkili problem, azınlık sınıfı (y = 1) için ağırlık $w$, model $q(x)\approx P(y=1\mid x)$. Ağırlıklı çapraz entropinin beklenen değeri, sabit bir $x$ için ($p=P(y=1\mid x)$):
$$\ell(q)=-w\,p\log q-(1-p)\log(1-q).$$
$\ell'(q)=-\dfrac{wp}q+\dfrac{1-p}{1-q}=0\ \Rightarrow\ q^*=\dfrac{wp}{wp+(1-p)}$, yani
$$\operatorname{logit}q^*=\log\frac{wp}{1-p}=\operatorname{logit}p+\log w .$$
**Sonuçlar:** (1) Esnek bir model ağırlıksız modelin logit'ini $\log w$ kadar yukarı kaydırır — 0.5 eşiğinde bu, eşiği $p=1/(1+w)$'ye indirmekle **aynıdır**. Ters frekans ağırlığı $w=(1-\pi)/\pi$ ile eşik tam öncel $\pi$'ye iner. (2) Tahmin edilen olasılıklar artık gerçek olasılıklar değildir (kalibrasyon bozulur); gerekirse çıkarımda logit'ten $\log w$ çıkararak geri al. (3) Beklenen değerde ağırlıklandırma ile azınlığı $w$ kat **fazla örneklemek** aynı kaybı verir; fark gradyan gürültüsündedir (fazla örnekleme aynı azınlık örneklerini tekrar tekrar gösterir, ağırlıklandırma nadir ama büyük gradyanlar üretir).
:::

Türetim "esnek model" varsayar. Kapasitesi sınırlı ya da yanlış belirlenmiş bir modelde ağırlıklandırma eşik kaydırmanın ötesine geçer: modelin kapasitesini azınlık bölgesine yönlendirir. Aşağıdaki deneyde lojistik regresyon doğru belirlenmiş olduğu için sabit terimin kayması log w'ye çok yakın çıkıyor; karar sınırı paralel kayıyor, recall artıyor, precision düşüyor.

::lab l3

### Veri setleri ve çerçeveler {#veri}

Derin ağların eğitimi eğitim verisinden büyük ölçüde etkilenir. Sık kullanılan bazı veri setleri:

| veri seti | içerik | boyut |
|---|---|---|
| MNIST | el yazısı rakamlar, 10 sınıf | 60 000 eğitim + 10 000 test, 28×28 gri |
| Fashion-MNIST | giysi görüntüleri (pantolon, spor ayakkabı, bot…), 10 sınıf | MNIST ile aynı biçim (yerine doğrudan konabilsin diye) |
| CIFAR-10 / 100 | doğal nesneler, 10 / 100 sınıf | 50 000 + 10 000, 32×32 renkli |
| ImageNet (ILSVRC) | 1000 sınıf | ≈ 1.28 milyon eğitim görüntüsü |
| COCO | nesne tespiti, segmentasyon, altyazı; 80 nesne kategorisi | ≈ 118 000 etiketli eğitim görüntüsü |

Verinin depolanması, yüklenmesi ve ön işlenmesi çoğu zaman kritiktir (GPU'nun veri beklememesi için); PyTorch'ta `Dataset`/`DataLoader` (M0.5), TensorFlow'da `tf.data` yalnız bu iş için vardır.

**Çerçeveler:** PyTorch (Meta), TensorFlow 2 + Keras ve JAX (Google) bugün yaygın olanlardır. MXNet (Amazon) ve CNTK (Microsoft) artık geliştirilmiyor; Caffe → Caffe2 → PyTorch'a katıldı. **ONNX** bir çerçeve değil, çerçeveler ve çalışma zamanları arasında model alışverişi için açık bir **biçimdir** (Microsoft, Meta, Intel, AMD, IBM, ARM…).

### Transfer öğrenme: önceden eğitilmiş katmanları yeniden kullanmak {#transfer}

Çok büyük bir derin ağı sıfırdan eğitmek genellikle iyi bir fikir değildir. Bunun yerine **benzer bir görevi** çözen mevcut bir ağ bul ve onun **alt katmanlarını** yeniden kullan: **transfer öğrenme**. Yalnız eğitimi büyük ölçüde hızlandırmaz, çok **daha az eğitim verisi** de gerektirir.

Örnek: hayvanlar, bitkiler, taşıtlar ve günlük nesneler dahil 100 kategoriyi sınıflayan eğitilmiş bir ağın var. Şimdi belirli **araç türlerini** sınıflayan bir ağ eğitmek istiyorsun. Görevler çok benzer, hatta kısmen örtüşüyor: ilk ağın parçalarını yeniden kullanmalısın.

**Neden işe yarar? Öznitelik hiyerarşisi.**
- **Alt katmanlar** (girdiye yakın) öznitelik çıkarıcılardır: kenarlar, renk lekeleri, dokular — neredeyse her görsel görevde işe yarar.
- **Orta–üst katmanlar** öznitelik birleştiricilerdir: parçalar, motifler, yüksek düzeyli öznitelikler.
- **Çıktı katmanları** çıkarılan ve birleştirilen öznitelikleri kullanarak karar veren sınıflandırıcılardır.

Transfer en iyi, girdilerin **benzer düşük düzeyli özniteliklere** sahip olduğu durumda çalışır. Yeni görevin görüntüleri orijinaldekiyle aynı boyutta değilse genelde bir ön işleme adımıyla orijinal modelin beklediği boyuta getirmek gerekir (ve aynı normalizasyon istatistikleri kullanılmalıdır).

<figure class="fig wide"><svg viewBox="0 0 560 258" role="img" aria-label="Önceden eğitilmiş ağın alt katmanları yeni göreve kopyalanır ve dondurulur, çıktı katmanı yenisiyle değiştirilir"><defs><marker id="tl1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><text x="90" y="18" text-anchor="middle" class="tp">A görevi: önceden eğitilmiş</text><rect x="20" y="30" width="140" height="26" rx="5" class="soft-b cb"/><text x="90" y="47" text-anchor="middle" class="sm">çıktı: 100 sınıf</text><rect x="20" y="64" width="140" height="26" rx="5" class="soft-a ca"/><text x="90" y="81" text-anchor="middle" class="sm">gizli 4 (üst)</text><rect x="20" y="98" width="140" height="26" rx="5" class="soft-a ca"/><text x="90" y="115" text-anchor="middle" class="sm">gizli 3</text><rect x="20" y="132" width="140" height="26" rx="5" class="soft-a ca"/><text x="90" y="149" text-anchor="middle" class="sm">gizli 2</text><rect x="20" y="166" width="140" height="26" rx="5" class="soft-a ca"/><text x="90" y="183" text-anchor="middle" class="sm">gizli 1 (alt)</text><rect x="20" y="200" width="140" height="26" rx="5" class="mute"/><text x="90" y="217" text-anchor="middle" class="sm">girdi</text><text x="370" y="18" text-anchor="middle" class="tp">B görevi: transfer</text><rect x="300" y="30" width="140" height="26" rx="5" class="soft-p cp"/><text x="370" y="47" text-anchor="middle" class="sm">yeni çıktı: 5 araç türü</text><text x="446" y="47" class="sm">eğitilir</text><rect x="300" y="64" width="140" height="26" rx="5" class="soft-a ca dash"/><text x="370" y="81" text-anchor="middle" class="sm">gizli 4</text><text x="446" y="81" class="sm">at / yeniden</text><rect x="300" y="98" width="140" height="26" rx="5" class="soft-a ca"/><text x="370" y="115" text-anchor="middle" class="sm">gizli 3</text><text x="446" y="115" class="sm">1. aşamada ❄</text><rect x="300" y="132" width="140" height="26" rx="5" class="soft-a ca"/><text x="370" y="149" text-anchor="middle" class="sm">gizli 2</text><text x="446" y="149" class="sm">❄ dondur</text><rect x="300" y="166" width="140" height="26" rx="5" class="soft-a ca"/><text x="370" y="183" text-anchor="middle" class="sm">gizli 1</text><text x="446" y="183" class="sm">❄ dondur</text><rect x="300" y="200" width="140" height="26" rx="5" class="mute"/><text x="370" y="217" text-anchor="middle" class="sm">girdi (yeniden boyutla)</text><line x1="162" y1="77" x2="296" y2="77" class="mute dash" marker-end="url(#tl1)"/><line x1="162" y1="111" x2="296" y2="111" class="mute dash" marker-end="url(#tl1)"/><line x1="162" y1="145" x2="296" y2="145" class="mute dash" marker-end="url(#tl1)"/><line x1="162" y1="179" x2="296" y2="179" class="mute dash" marker-end="url(#tl1)"/><text x="230" y="248" text-anchor="middle" class="sm">ağırlıkları kopyala (alt katmanlar genel öznitelik çıkarıcıdır)</text></svg><figcaption>Şekil 2 — Önceden eğitilmiş katmanları yeniden kullanmak. Çıktı katmanı neredeyse her zaman değiştirilir (yeni görevin sınıf sayısı farklı); üst gizli katmanlar daha göreve özgüdür, alt katmanlar daha geneldir. Görevler ne kadar benzerse o kadar çok katman yeniden kullanılır. Sıra: dondur → yeni başlığı eğit → üst katmanları çöz → küçük öğrenme oranıyla ince ayar.</figcaption></figure>

**Kurallar (deneme yanılmaya dayalı sezgisel yöntemler — sihirli bir tarif yok):**
1. Orijinal modelin **çıktı katmanını değiştir**: büyük olasılıkla yeni görev için hiç yararlı değildir, doğru sayıda çıktısı bile olmayabilir.
2. Orijinal modelin **üst gizli katmanları** alt katmanlar kadar yararlı değildir: yeni görev için en yararlı yüksek düzeyli öznitelikler orijinal görevdekilerden çok farklı olabilir.
3. Görevler ne kadar **benzerse** (alt katmanlardan başlayarak) o kadar çok katman yeniden kullan. Çok benzer görevlerde tüm gizli katmanları tut, yalnız çıktı katmanını değiştir.
4. Önce yeniden kullanılan tüm katmanları **dondur** (eğitilemez yap), yalnız yeni başlığı eğit — rastgele başlatılmış başlığın büyük ilk gradyanları önceden eğitilmiş ağırlıkları bozmasın. Performansa bak.
5. Sonra üstteki bir–iki gizli katmanı **çöz**, geri yayılımın onları ayarlamasına izin ver; **öğrenme oranını küçült** (ör. 10 kat), iyi ağırlıklar bozulmasın. Ne kadar çok eğitim verin varsa o kadar çok katman çözebilirsin.
6. Hâlâ iyi değilse ve az veri varsa: üst gizli katmanları atmayı ve kalanları yeniden dondurmayı dene; çok veri varsa üst katmanları atmak yerine değiştirmeyi, hatta daha çok gizli katman eklemeyi dene.

:::ayrıntı Yosinski ve ark. (2014): öznitelikler ne kadar aktarılabilir?
ImageNet'i rastgele iki yarıya (A, B) bölüp katman katman aktarım denediler. Bulgular: (1) aktarılabilirlik görevler arası mesafe arttıkça azalır ve üst katmanlar göreve daha özgüdür; (2) **orta** katmanlardan bölmek, donmuş katmanlarla yeni katmanlar arasındaki "kırılgan ortak uyumu" (co-adaptation) bozduğu için ek bir optimizasyon zorluğu yaratır; (3) aktarılan ağırlıklarla başlatıp **tüm ağı ince ayarlamak**, büyük hedef veride bile rastgele başlatmadan **daha iyi genelleyen** ağlar verdi — başlangıcın etkisi uzun eğitimden sonra da kalıcı.
:::

:::uyarı PyTorch'ta dondurma ve BatchNorm
`for p in model.features.parameters(): p.requires_grad = False` yalnız ağırlıkları dondurur. BatchNorm'un **koşu istatistikleri** `model.train()` modunda güncellenmeye devam eder; küçük ve farklı dağılımlı yeni veriyle bozulabilirler. Donmuş kısımdaki BN katmanlarını ayrıca `eval()` moduna al (M3.6). Optimizer'a yalnız `requires_grad=True` parametreleri ver; çözdükten sonra optimizer'ı yeniden kur ya da parametre grubunu ekle (farklı öğrenme oranlarıyla: alt katmanlara küçük, başlığa büyük — "ayrımlı öğrenme oranı").
:::

### Gözetimsiz ön eğitim {#gozetimsiz}

Şimdiye kadarki tüm örnekler **gözetimliydi**. Peki şu durumda ne yaparız: (1) gözetimli eğitimi sıfırdan yapacak kadar etiketli verimiz yok, (2) benzer bir görevde eğitilmiş bir model de bulamıyoruz, dolayısıyla transfer uygulayamıyoruz — ama (3) bol **etiketsiz** verimiz var.

Gözetimsiz öğrenme alternatif bir yaklaşımdır; bugün gözetimsiz (ve öz-gözetimli) teknikler test performansında gözetimli öğrenmeye giderek yaklaşıyor. Ama ayrıca **tamamlayıcı bir ön eğitim aşaması** olarak da kullanılabilir: etiketsiz veriyle kabul edilebilir bir başlangıç noktasına ulaşılır, sonra az sayıdaki etiketli veriyle gözetimli ince ayara geçilir. Tarihsel olarak derin ağları eğitmeyi ilk mümkün kılan da buydu: Hinton ve ark. (2006) katmanları **tek tek, açgözlü biçimde** gözetimsiz (kısıtlı Boltzmann makineleri, sonra otokodlayıcılar) eğitip üst üste koydu; bugün bunun yerine tüm ağ tek seferde öz-gözetimli bir görevle eğitiliyor.

:::örnek Hayali bir vaka çalışması (ders slaytı) — ve cevabı
**Kurulum:** sınıf sayısı bilinen bir görsel sınıflandırma problemi; çok az etiketli veri, benzer görevde eğitilmiş model yok, ama çok etiketsiz veri. Ağ yapısına deneyimle karar verdin.

**Yöntem:** etiketsiz veriyi ağdan geçir, öznitelikleri al (son sınıflandırıcı katman(lar) hariç çıktı). Ham öznitelikler üzerinde **k-means** çalıştır. Küme etiketlerini girdilere sanki gerçek sınıf etiketleriymiş gibi ata. Bu etiketlerle birkaç epoch SGD + geri yayılım ile tüm modeli (sınıflandırıcı dahil) eğit. Öznitelikleri yeniden çıkar, yeniden kümele… Süreç yakınsayınca az sayıdaki etiketli veriyle gözetimli eğitime geç.

**Neden gözetimsiz ön eğitim?** Hiçbir insan etiketi kullanılmıyor: etiketler verinin kendi yapısından (kümelerden) geliyor. **Ne olur?** Bu fikir gerçekten çalışır — **DeepCluster** (Caron ve ark. 2018) tam olarak bunu yaptı ve ImageNet'te etiketsiz eğitilen ağın öznitelikleri başka görevlere iyi aktarıldı. Rastgele başlatılmış bir CNN'in öznitelikleri bile k-means'e anlamlı bir başlangıç verecek kadar yapı taşır. Ama **önemsiz çözümlere** dikkat: ağ her şeyi tek bir kümeye eşlerse (ya da birkaç dev küme) kayıp kolayca küçülür ve hiçbir şey öğrenilmez. DeepCluster bunu boşalan kümeleri yeniden başlatarak ve eğitim örneklerini kümelere göre **eşit** örnekleyerek önledi; ayrıca her kümeleme turunda küme numaraları değiştiği için sınıflandırıcı katmanı her turda yeniden başlatılmalıdır.
:::

### Yardımcı görevle ön eğitim ve öz-gözetimli öğrenme {#ozgozetim}

Verinin etiketsiz olması gözetimli öğrenmede kullanılamayacağı anlamına gelmez: etiketler **verinin kendisinden** de elde edilebilir. Buna **yardımcı bir görevle ön eğitim** denir. Ana fikir: etiketsiz verinin bazı parçalarını çıkarmak ve modeli eksik parçaları tamamlamaya eğitmek gibi bir yardımcı görev bulmak.

Az etiketli verin varsa: (1) etiketli eğitim verisini kolayca elde edebildiğin ya da üretebildiğin **yardımcı bir görevde** ilk ağı eğit; (2) o ağın alt katmanlarını asıl görevin için yeniden kullan. İlk ağın alt katmanları, ikinci ağın da büyük olasılıkla yeniden kullanabileceği öznitelik dedektörlerini öğrenir.

**Örnek (doğal dil işleme):** milyonlarca belgelik bir derlem indir ve etiketli veriyi otomatik üret: bazı kelimeleri rastgele maskele ve modeli eksik kelimeyi tahmin etmeye eğit. "What ___ you saying?" cümlesindeki eksik kelimenin büyük olasılıkla "are" ya da "were" olduğunu tahmin etmeli. Model bu görevde iyi performansa ulaşırsa dil hakkında zaten çok şey biliyordur; onu asıl görevin için yeniden kullanıp etiketli verinle ince ayar yapabilirsin. BERT (Devlin ve ark. 2019) tam olarak bunu yapar (token'ların %15'ini maskeler); GPT ailesinin "sonraki token'ı tahmin et" görevi de bir yardımcı görevdir (M4.5, M4.7).

:::tanım Öz-gözetimli öğrenme (self-supervised learning)
Etiketlerin otomatik olarak verinin kendisinden üretildiği, ardından modelin bu "etiketli" veri setinde **gözetimli öğrenme teknikleriyle** eğitildiği yaklaşım. Hiçbir insan etiketi gerektirmediği için en doğru biçimde bir **gözetimsiz öğrenme** türü olarak sınıflanır.
:::

Görüntüde yaygın öz-gözetimli görevler: maskelenmiş yamaları tamamlamak (MAE), aynı görüntünün iki farklı artırılmış görünümünün temsillerini birbirine yaklaştırıp diğer görüntülerinkinden uzaklaştırmak (karşıtsal öğrenme, SimCLR), renklendirme, döndürme açısını tahmin etmek. Ortak fikir: görevi çözmek için verinin **anlamsal yapısını** öğrenmek zorunda kalacağı, ama etiketi bedava olan bir görev bulmak.

### Özet {#ozet}

- Teşhis iki sayıyla başlar: eğitim hatası yüksekse yanlılık, eğitim–doğrulama farkı büyükse varyans. Öğrenme eğrisi "daha çok veri yardım eder mi?" sorusunu yanıtlar.
- Hiperparametrelerde önce öğrenme oranı; ızgara yerine log ölçekte rastgele arama, kabadan inceye, kötü denemeleri erken kes.
- Üç küme: eğitim, doğrulama (seçim), test (yalnız en sonda, bir kez). Test kümesine göre yapılan her seçim onu doğrulama kümesine çevirir.
- Sızıntı türleri: ön işleme, grup, zaman, hedef. Hepsinin ortak çaresi: değerlendirme protokolünü tahmin anındaki gerçek durumu taklit edecek biçimde kurmak.
- Sınıf ağırlığı $w$ esnek bir modelde logit'i $\log w$ kaydırır (eşik kaydırmaya denk), kalibrasyonu bozar; doğruluk yerine recall/precision/dengeli doğruluk ölç.
- Transfer: çıktı katmanını değiştir, alt katmanları dondur, başlığı eğit, sonra üst katmanları küçük öğrenme oranıyla çöz; görev benzerliği kaç katman kullanılacağını belirler.
- Etiket yoksa: gözetimsiz ön eğitim (ör. DeepCluster — önemsiz çözümlere dikkat) ya da öz-gözetimli yardımcı görev (maskeli dil modeli, sonraki token, karşıtsal öğrenme).
