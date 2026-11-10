### GAN eğitimi neden zor? Nash dengesi {#nash}

Eğitim sırasında üretici ve ayırıcı, sıfır toplamlı bir oyunda sürekli birbirini alt etmeye çalışır. Eğitim ilerledikçe oyun, oyun kuramcılarının **Nash dengesi** dediği (matematikçi John Nash'in adıyla) bir duruma varabilir: diğer oyuncular stratejilerini değiştirmezse hiçbir oyuncunun kendi stratejisini değiştirmekle daha iyi duruma geçemeyeceği durum.

Slayttaki örnek: herkes yolun solundan gidiyorsa bu bir Nash dengesidir; tek başına sağa geçen sürücü daha iyi duruma geçmez. Elbette ikinci bir Nash dengesi de vardır: herkesin sağdan gitmesi. Farklı başlangıç durumları ve dinamikler bir dengeye ya da ötekine götürebilir. Bu örnekte denge kurulunca tek bir en iyi strateji vardır (herkesle aynı taraftan sürmek). Ama bir Nash dengesi birbiriyle yarışan birden çok strateji de içerebilir: avcı avını kovalar, av kaçmaya çalışır ve ikisi de stratejisini değiştirmekle kazanmaz.

GAN'lara gelince: özgün GAN makalesinin yazarları bir GAN'ın yalnız tek bir Nash dengesine ulaşabileceğini gösterdi. Bu, üreticinin kusursuz gerçekçi görüntüler ürettiği ve ayırıcının tahmine zorlandığı (%50 gerçek, %50 sahte) durumdur. Bu sonuç cesaret vericidir: GAN'ı yeterince uzun eğitmek yetecek ve sonunda bu dengeye varılacakmış gibi görünür. Ne yazık ki o kadar basit değildir: dengeye bir gün ulaşılacağını hiçbir şey garanti etmez.

:::uyarı Kaynak notu: "tek Nash dengesi" sonucunun koşulları
M5.2'de türetildiği gibi bu sonuç **fonksiyon uzayında** ve sonsuz kapasite varsayımıyla geçerlidir: en iyi ayırıcı için $C(G)=-\log4+2\,\mathrm{JS}(p_{data}\|p_g)$ ve en küçüğü yalnız $p_g=p_{data}$'da. Gerçek ağlarda optimizasyon parametre uzayında yapılır, oyun dışbükey–içbükey değildir ve birçok yerel denge (ya da hiç yakınsamayan çevrimler) olabilir. M5.2'deki Dirac GAN, en basit durumda bile eşzamanlı gradyan adımlarının dengenin etrafında dışa doğru döndüğünü gösterir.
:::

### Mod çöküşü ve kararsızlık {#cokus}

En büyük zorluğa **mod çöküşü** (mode collapse) denir: üreticinin çıktılarının giderek daha az çeşitli hâle gelmesi. Bu nasıl olabilir? Slayttaki Fashion MNIST senaryosunu adım adım izleyelim:
1. Üretici, ayakkabı üretmede diğer sınıflardan daha iyi hâle gelsin.
2. Ayakkabılarla ayırıcıyı biraz daha çok kandırır; bu da onu daha çok ayakkabı üretmeye teşvik eder.
3. Yavaş yavaş başka bir şey üretmeyi unutur.
4. Bu arada ayırıcının gördüğü tek sahte görüntüler ayakkabılardır; o da diğer sınıflardaki sahte görüntüleri ayırt etmeyi unutur.
5. Sonunda ayırıcı sahte ayakkabıları gerçeklerinden ayırmayı başarınca üretici başka bir sınıfa kaymak zorunda kalır.
6. Bu kez gömleklerde iyileşebilir, ayakkabıları unutur; ayırıcı da onu izler.

GAN birkaç sınıf arasında yavaş yavaş dolaşabilir ve hiçbirinde gerçekten iyi olamaz. M5.2'deki 1-B GAN deneyinde (gan1d) tek tepeli bir üreticinin iki tepe arasında gidip gelmesi bu çevrimin en küçük modelidir.

Ayrıca üretici ve ayırıcı sürekli birbirine karşı ittiği için parametreleri salınabilir ve kararsız hâle gelebilir. Eğitim düzgün başlayıp bu kararsızlıklar yüzünden görünür bir neden olmadan aniden ıraksayabilir. Bu karmaşık dinamikleri birçok etken etkilediği için GAN'lar hiperparametrelere çok duyarlıdır.

:::sezgi Neden "unutma"?
Ayırıcı her adımda yalnız **şu anki** üreticinin örneklerini görür. Üretici tek bir moda çöktüyse ayırıcının kaybı diğer bölgeler hakkında hiçbir şey söylemez; oradaki karar sınırı serbestçe kayar. Üretici geri döndüğünde ayırıcı orayı "unutmuş" olur. Bu, sürekli öğrenmedeki felaket unutmanın (catastrophic forgetting) bir biçimidir. Aşağıdaki iki çarenin ortak fikri, ayırıcıya şu anki batch'ten daha geniş bir resim göstermektir.
:::

### Çareler {#careler}

Bu sorunlar 2014'ten beri araştırmacıları çok meşgul etti. Bu konuda çok sayıda makale yayımlandı; bazıları eğitimi kararlı kılmak ya da mod çöküşünden kaçınmak için yeni maliyet fonksiyonları ya da teknikler önerdi.

- **Deneyim tekrarı** (experience replay). Üreticinin her yinelemede ürettiği görüntüleri bir tekrar tamponunda saklamak (eski görüntüler yavaş yavaş atılır) ve ayırıcıyı gerçek görüntüler artı bu tampondan çekilen sahte görüntülerle eğitmek (yalnız şu anki üreticinin ürettikleriyle değil). Bu, ayırıcının son üreticinin çıktılarına aşırı uyum sağlama olasılığını azaltır. (GAN'larda bu fikri Shrivastava ve ark. 2017 "geçmiş tamponu" adıyla kullandı; ad, pekiştirmeli öğrenmedeki DQN'den gelir.)
- **Mini-batch ayrımı** (minibatch discrimination; Salimans ve ark. 2016). Görüntülerin batch içinde birbirine ne kadar benzediğini ölçer ve bu istatistiği ayırıcıya verir. Böylece ayırıcı, çeşitlilikten yoksun bir sahte görüntü batch'ini tamamen kolayca reddedebilir. Bu, üreticiyi daha çeşitli görüntüler üretmeye teşvik eder ve mod çöküşü olasılığını azaltır. Somut olarak her örneğin ara özniteliği bir $\mathbf M_i\in\mathbb R^{B\times C}$ matrisine izdüşürülür ve $o(\mathbf x_i)_b=\sum_j\exp(-\|\mathbf M_{i,b}-\mathbf M_{j,b}\|_1)$ ayırıcıya ek öznitelik olarak verilir. Bütün batch aynı noktaya çökerse bu toplam büyür ve ayırıcı bunu görür. ProGAN'ın mini-batch standart sapma katmanı (aşağıda) bu fikrin parametresiz basitleştirmesidir.
- Diğer makaleler yalnızca iyi çalışan özel mimariler önerir (DCGAN, aşağıda).

Aynı makale (Salimans ve ark. 2016) **öznitelik eşleme** (üreticinin, ayırıcının ara katmanındaki gerçek veri ortalamasını tutturması) ve tek taraflı etiket yumuşatmayı da önerir. FID'i tanıtan makale (Heusel ve ark. 2017) ise iki ağa farklı öğrenme oranı vermeyi (iki zaman ölçekli güncelleme kuralı, TTUR) önerip bunun yerel bir Nash dengesine yakınsadığını kanıtlar.

**Maliyet fonksiyonunu iyileştirmek** (slayt 83). Eğitimin başında $D(G(z))\approx0$'dır ve özgün kayıp $\log(1-D(G(z)))$'nin orada türevi çok küçüktür; üretici öğrenemez. Değiştirilmiş (doymayan) kayıp $-\log D(G(z))$ bu bölgede büyük gradyan verir. Türetimi ve iki eğrinin karşılaştırması M5.2'dedir.

Kısacası bu hâlâ çok etkin bir araştırma alanıdır ve GAN'ların dinamikleri hâlâ tam olarak anlaşılmış değildir. Ama iyi haber, büyük ilerleme kaydedilmiş olmasıdır.

### Derin konvolüsyonel GAN'lar (DCGAN) {#dcgan}

Özgün GAN makalesinin yazarları konvolüsyon katmanlarıyla deney yaptı ama yalnız küçük görüntüler üretmeyi denedi. Kısa süre sonra birçok araştırmacı büyük görüntüler için daha derin konvolüsyonel ağlara dayalı GAN'lar kurmaya çalıştı. Eğitim çok kararsız olduğu için bu zor oldu, ama Alec Radford ve arkadaşları birçok farklı mimari ve hiperparametreyle deney yaptıktan sonra 2015'in sonunda başardı. Mimarilerine **derin konvolüsyonel GAN** (DCGAN) adını verdiler. Kararlı konvolüsyonel GAN'lar kurmak için önerdikleri ana kurallar:
- Havuzlama katmanlarını adımlı konvolüsyonlarla (ayırıcıda) ve transposed konvolüsyonlarla (üreticide) değiştir.
- Üreticinin çıktı katmanı ve ayırıcının girdi katmanı dışında, iki ağda da batch normalizasyonu kullan.
- Daha derin mimarilerde tam bağlı gizli katmanları kaldır.
- Üreticide çıktı katmanı (tanh) dışındaki tüm katmanlarda ReLU kullan.
- Ayırıcıda tüm katmanlarda LeakyReLU kullan.

<figure class="fig wide"><svg viewBox="0 0 640 310" role="img" aria-label="DCGAN: üretici z'den 4×4, 8×8 ve 16×16'ya transposed konvolüsyonlarla büyür; ayırıcı 16×16'dan adımlı konvolüsyonlarla tek bir logite iner"><defs><marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker><marker id="a2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker></defs><text x="10" y="18" class="tb">Üretici (198 017 parametre)</text><rect x="20" y="61.0" width="60" height="18" rx="4" class="soft-b"/><text x="50.0" y="74.0" text-anchor="middle" class="sm">z</text><rect x="175" y="55.0" width="110" height="30" rx="4" class="soft-b"/><text x="230.0" y="74.0" text-anchor="middle" class="sm">128 × 4 × 4</text><rect x="330" y="47.0" width="110" height="46" rx="4" class="soft-b"/><text x="385.0" y="74.0" text-anchor="middle" class="sm">64 × 8 × 8</text><rect x="500" y="31.0" width="110" height="78" rx="4" class="soft-a"/><text x="555.0" y="74.0" text-anchor="middle" class="sm">1 × 16 × 16</text><path d="M84,70 L171,70" class="cb" marker-end="url(#a1)"/><text x="127.5" y="50" text-anchor="middle" class="sm">ConvT 4, s1 · BN · ReLU</text><path d="M289,70 L326,70" class="cb" marker-end="url(#a1)"/><text x="307.5" y="120" text-anchor="middle" class="sm">ConvT 4, s2 · BN · ReLU</text><path d="M444,70 L496,70" class="cb" marker-end="url(#a1)"/><text x="470.0" y="120" text-anchor="middle" class="sm">ConvT 4, s2 · tanh</text><text x="10" y="170" class="ta">Ayırıcı (134 465 parametre)</text><rect x="20" y="183.0" width="110" height="78" rx="4" class="soft-a"/><text x="75.0" y="226.0" text-anchor="middle" class="sm">1 × 16 × 16</text><rect x="175" y="199.0" width="110" height="46" rx="4" class="soft-p"/><text x="230.0" y="226.0" text-anchor="middle" class="sm">64 × 8 × 8</text><rect x="330" y="207.0" width="110" height="30" rx="4" class="soft-p"/><text x="385.0" y="226.0" text-anchor="middle" class="sm">128 × 4 × 4</text><rect x="500" y="213.0" width="60" height="18" rx="4" class="soft-b"/><text x="530.0" y="226.0" text-anchor="middle" class="sm">logit</text><path d="M134,222 L171,222" class="ca" marker-end="url(#a2)"/><text x="152.5" y="272" text-anchor="middle" class="sm">Conv 4, s2 · LReLU</text><path d="M289,222 L326,222" class="ca" marker-end="url(#a2)"/><text x="307.5" y="272" text-anchor="middle" class="sm">Conv 4, s2 · BN · LReLU</text><path d="M444,222 L496,222" class="ca" marker-end="url(#a2)"/><text x="470.0" y="272" text-anchor="middle" class="sm">Conv 4, s1</text><text x="10" y="300" class="sm">Havuz yok (adımlı/transposed konvolüsyon), tam bağlı gizli katman yok; BN üretici çıkışında ve ayırıcı girişinde yok.</text></svg><figcaption>Şekil 1 — Aşama d'deki DCGAN (16 × 16 rakamlar). Kutu yükseklikleri uzamsal boyutla orantılı. Radford ve ark.'nın kuralları: üreticide ReLU ve tanh çıkış, ayırıcıda LeakyReLU, iki ağda da BN (üretici çıkışı ve ayırıcı girişi hariç).</figcaption></figure>

:::sezgi Kuralların gerekçesi
- **Havuz yerine öğrenilen örnekleme**: üretici büyütmeyi, ayırıcı küçültmeyi kendisi öğrenir; maks havuzun seyrek gradyanı (yalnız en büyük elemana) yerine her konuma gradyan akar.
- **BN**: katmanların girdi ölçeğini sabitler ve başlangıçta üreticinin tek bir noktaya çökmesini zorlaştırır. Üretici çıkışında BN olsaydı çıktının ortalaması ve ölçeği veriye değil batch'e bağlanırdı; ayırıcı girişinde olsaydı gerçek ve sahte batch'lerin istatistikleri ayrı ayrı normalize edilir ve fark silinirdi. (Makale, tüm katmanlara BN koymanın örnek salınımına ve kararsızlığa yol açtığını bildirir.)
- **tanh çıkışı**: veri $[-1,1]$'e ölçeklenir; çıkış sınırlıdır ve doyuma yaklaşınca da gradyan akar.
- **LeakyReLU ayırıcıda**: negatif bölgede de gradyan geçer; üreticiye giden sinyal ayırıcının ölü birimlerinde kaybolmaz.
:::

**El hesabı.** Üreticinin 4 × 4 haritası, çekirdeği 4, adımı 2, dolgusu 1 olan bir transposed konvolüsyondan geçerse kenar $(4-1)\cdot2-2\cdot1+4=\mathbf8$ olur (M4.10'daki formül). Bu (4, 2, 1) üçlüsü kenarı tam iki katına çıkardığı için DCGAN'ların standart bloğudur.

Bu kurallar çoğu durumda işe yarar ama her zaman değil; hâlâ farklı hiperparametrelerle deney yapmak gerekebilir. Hatta aynı modeli yalnız rastgele tohumu değiştirerek yeniden eğitmek bazen işe yarar.

### Ne zaman durmalı? Başarım ölçütleri {#metrik}

Eğitimi ne zaman durdurmalı? Daha önce söylediğimiz gibi Nash dengesine ulaşılması garanti değildir. Üstelik eğitimi hızlandırmak için genellikle değiştirilmiş kaybı kullanırız; bu durumda kayıp eğrileri de iyi bir ölçüt değildir. Bir yol, üretici çıktılarından örnekleyip sonuçlara bakmaktır. Örneklerin ne kadar iyi olduğunu ölçen görsel başarım ölçütleri de kullanılabilir.

VAE'nin aksine GAN **örtük** (implicit) bir veri dağılımı kullanır: örneklenen çıktının olasılığını (olabilirliğini) vermez. Bu yüzden sahte örneklerin kalitesini ölçmek doğrudan değildir. Bu amaçla iki ölçüt yaygındır: Inception Score ve Fréchet Inception Distance.

**Inception Score (IS).** ImageNet'te (1000 sınıf) önceden eğitilmiş bir Inception v3 modeliyle sahte görüntüler değerlendirilir. İki ölçüt vardır:
- **Görüntü kalitesi**: örneklenen sahte görüntüler belirli bir nesneye benziyor mu? Verilen bir $z$ için softmax çıktısı $p(y\mid x)$ düşük entropili olmalıdır (tek bir sınıfa yüksek güven).
- **Görüntü çeşitliliği**: model çok çeşitli nesnelerden örnekleyebiliyor mu? Birçok $z$ üzerinden **ortalama** sınıf dağılımı $p(y)=\mathbb E_x\,p(y\mid x)$ yüksek entropili olmalıdır.

:::türetim IS = exp(karşılıklı bilgi)
Tanım: $\mathrm{IS}=\exp\big(\mathbb E_x\,\mathrm{KL}(p(y\mid x)\,\|\,p(y))\big)$. KL'yi açalım:
$$\mathbb E_x\sum_yp(y\mid x)\log\frac{p(y\mid x)}{p(y)}=\underbrace{\mathbb E_x\sum_yp(y\mid x)\log p(y\mid x)}_{-\mathbb E_x H(y\mid x)}-\sum_y\underbrace{\mathbb E_x[p(y\mid x)]}_{p(y)}\log p(y)=H(y)-\mathbb E_xH(y\mid x).$$
Bu, $x$ ile $y$ arasındaki karşılıklı bilgidir, $I(x;y)$. Slayttaki iki ölçüt tam olarak bu iki terimdir: kalite $\mathbb E H(y\mid x)$'i küçültür, çeşitlilik $H(y)$'yi büyütür. Sınırlar: $0\le I\le H(y)\le\log C$, dolayısıyla $1\le\mathrm{IS}\le C$ (ImageNet'te 1000).

El hesabı: örneklerin yarısında $p(y\mid x)=(0.9,0.1)$, yarısında $(0.1,0.9)$. $p(y)=(0.5,0.5)$ ve her örnek için $\mathrm{KL}=0.9\ln1.8+0.1\ln0.2=0.529-0.161=0.368$, dolayısıyla $\mathrm{IS}=e^{0.368}=\mathbf{1.445}$ (en çok 2).
:::

**Fréchet Inception Distance (FID).** Yine eğitilmiş Inception v3 kullanılır, ama çıktı olasılıkları yerine son havuzlama katmanının çıktısındaki öznitelik haritaları (2048 boyut) kullanılarak gerçek ve sahte görüntülerin öznitelik dağılımları karşılaştırılır. Gerçek ve sahte özniteliklerin çok değişkenli Gauss dağıldığı varsayılır ve iki Gauss arasındaki uzaklık Fréchet uzaklığıyla hesaplanır:
$$\mathrm{FID}=\|\boldsymbol\mu_r-\boldsymbol\mu_g\|^2+\mathrm{tr}\big(\boldsymbol\Sigma_r+\boldsymbol\Sigma_g-2(\boldsymbol\Sigma_r\boldsymbol\Sigma_g)^{1/2}\big).$$
FID ne kadar düşükse örnek kalitesi o kadar iyidir.

:::türetim Bir boyutta Fréchet uzaklığı
Fréchet uzaklığı (2-Wasserstein uzaklığı), iki dağılımı birleştiren tüm eşleştirmeler (coupling) üzerinden $\min\mathbb E\,(X-Y)^2$'dir. Bir boyutta en iyi eşleştirme **monoton** olandır: $X$'in $q$-yüzdeliği $Y$'nin $q$-yüzdeliğine eşlenir. İki Gauss için bu doğrusal bir haritadır, $Y=\mu_2+\frac{\sigma_2}{\sigma_1}(X-\mu_1)$. Yerine koyalım ($X=\mu_1+\sigma_1\varepsilon$):
$$X-Y=(\mu_1-\mu_2)+(\sigma_1-\sigma_2)\varepsilon\;\Rightarrow\;\mathbb E(X-Y)^2=(\mu_1-\mu_2)^2+(\sigma_1-\sigma_2)^2 .$$
Bu, genel formülün 1-B hâlidir: $\sigma_1^2+\sigma_2^2-2\sigma_1\sigma_2=(\sigma_1-\sigma_2)^2$. El hesabı: $\mathcal N(0,1)$ ve $\mathcal N(1,2^2)$ için $1+1=\mathbf2$. Çok boyutta, kovaryanslar köşegense her eksen ayrı katkı verir: $\boldsymbol\mu$ farkı $(1,0)$, $\boldsymbol\Sigma_r=\mathbf I$, $\boldsymbol\Sigma_g=\mathrm{diag}(4,9)$ için $1+(1-2)^2+(1-3)^2=\mathbf6$. Genel durumda iz teriminin matris karekökü gerekir (aşama c: öz ayrışım ya da Newton–Schulz).
:::

:::uyarı Kaynak notu: slayt 89–90'daki tanımlar
Slayt 89'daki "tüm örneklenmiş görüntülerin entropisi yüksek olmalı" ifadesi, tek tek görüntülerin değil, **ortalama sınıf dağılımı** $p(y)$'nin entropisidir; tek görüntülerin entropisi tam tersine düşük olmalıdır. Slayt 90'daki "düşük FID = iyi" doğrudur, ama FID'in iki önemli zayıflığı vardır. (1) Yanlı bir kestiricidir, değeri örnek sayısına bağlıdır; farklı $N$'lerle hesaplanmış FID'ler karşılaştırılamaz (Chong ve Forsyth 2020; ek x2'de FID∞). (2) Tek bir Gauss ile özetlediği için kalite ile çeşitliliği tek bir sayıda karıştırır. Bunları ayırmak için kesinlik ve geri çağırma (Kynkäänniemi ve ark. 2019) önerildi; aşağıdaki ikinci deneyde kullanılıyor.
:::

Aşağıdaki deneyde 10 sınıflı sentetik bir "öznitelik uzayında" üreticiyi bozarak IS ile FID'in neye duyarlı olduğunu gör. Sınıf başına tek örnek üreten (çeşitliliği yok) bir üreticinin IS'i gerçek veriden **yüksek** çıkar; FID ise bunu yakalar. Öznitelik boyutunu artırıp $N$'yi azaltınca FID'in yanlılığını izle.

::lab l1

### Progresif büyüyen GAN'lar (ProGAN) {#progan}

2018'deki bir makalede Nvidia araştırmacıları Tero Karras ve arkadaşları önemli bir teknik önerdi: eğitimin başında küçük görüntüler üretmek, sonra hem üreticiye hem ayırıcıya yavaş yavaş konvolüsyon katmanları ekleyerek giderek daha büyük görüntüler üretmek (4 × 4, 8 × 8, 16 × 16, …, 512 × 512, 1024 × 1024). Bu yaklaşım yığılmış otomatik kodlayıcıların açgözlü katman katman eğitimine benzer (M5.6). Ek katmanlar üreticinin sonuna ve ayırıcının başına eklenir; daha önce eğitilmiş katmanlar eğitilebilir kalır.

:::uyarı Kaynak düzeltmesi: yazar adı
Slayt 91'de "Tero Kerras" yazıyor; doğrusu **Tero Karras**. Makale arXiv'e 2017'de çıktı, ICLR 2018'de yayımlandı.
:::

Yeni katman eklenince ağ bir anda bozulmasın diye yeni katman **yumuşak geçişle** (fade-in) girer:

<figure class="fig wide"><svg viewBox="0 0 640 250" role="img" aria-label="ProGAN yumuşak geçişi: eski düşük çözünürlüklü yolun büyütülmüş çıktısı (1 − α) ile, yeni katmanın çıktısı α ile karıştırılır; α geçiş süresince 0'dan 1'e doğrusal artar"><defs><marker id="f1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="20" y="30" width="120" height="34" rx="4" class="soft-b"/><text x="80.0" y="51.0" text-anchor="middle" class="sm">16 × 16 blokları</text><rect x="200" y="10" width="110" height="30" rx="4" class="soft-b"/><text x="255.0" y="29.0" text-anchor="middle" class="sm">toRGB (eski)</text><rect x="350" y="10" width="110" height="30" rx="4" class="soft-b"/><text x="405.0" y="29.0" text-anchor="middle" class="sm">2× büyüt</text><rect x="200" y="80" width="110" height="30" rx="4" class="soft-a"/><text x="255.0" y="99.0" text-anchor="middle" class="sm">yeni 32 × 32 blok</text><rect x="350" y="80" width="110" height="30" rx="4" class="soft-a"/><text x="405.0" y="99.0" text-anchor="middle" class="sm">toRGB (yeni)</text><path d="M140,40 L196,25" class="ink" marker-end="url(#f1)"/><path d="M140,55 L196,92" class="ink" marker-end="url(#f1)"/><path d="M310,25 L346,25" class="ink" marker-end="url(#f1)"/><path d="M310,95 L346,95" class="ink" marker-end="url(#f1)"/><circle cx="530" cy="60" r="16" class="ink"/><text x="530" y="65" text-anchor="middle">+</text><path d="M460,25 L516,50" class="ink" marker-end="url(#f1)"/><path d="M460,95 L516,70" class="ink" marker-end="url(#f1)"/><text x="480" y="22" class="tb">× (1 − α)</text><text x="480" y="112" class="ta">× α</text><path d="M546,60 L600,60" class="ink" marker-end="url(#f1)"/><text x="600" y="50" text-anchor="end" class="sm">32 × 32</text><path d="M60,220 L580,220" class="mute"/><path d="M60,150 L60,220" class="mute"/><path d="M60,220 L320.0,150 L580,150" class="ca w2"/><text x="54" y="154" text-anchor="end" class="sm">α = 1</text><text x="54" y="220" text-anchor="end" class="sm">0</text><text x="190.0" y="236" text-anchor="middle" class="sm">geçiş: α doğrusal artar</text><text x="450.0" y="236" text-anchor="middle" class="sm">kararlılaştırma: yalnız yeni yol</text><circle cx="138.0" cy="199.0" r="4" class="cb-f"/><text x="146.0" y="213.0" class="sm">α = 0.3: 0.7 · 2 + 0.3 · 4 = 2.6</text></svg><figcaption>Şekil 2 — ProGAN'da yeni çözünürlüğün eklenmesi. Yeni katman ağı bir anda bozmasın diye çıktı iki yolun karışımıdır; α 0'dan 1'e çıkınca eski yol devreden çıkar. Ayırıcıda ayna simetrik bir geçiş yapılır (fromRGB ve 2× küçültme). Sayılar modülün el hesabı alanından.</figcaption></figure>

Makale ayrıca çıktıların çeşitliliğini artırmayı (mod çöküşünden kaçınmayı) ve eğitimi daha kararlı kılmayı amaçlayan başka teknikler de tanıttı.
- **Mini-batch standart sapma katmanı** (ayırıcının sonuna yakın). Her öznitelik ve her konum için batch üzerindeki standart sapma hesaplanır, hepsinin ortalaması alınarak tek bir sayı elde edilir. Bu sayı, her örneğe sabit bir ek öznitelik haritası olarak eklenir. Üretici az çeşitli görüntüler üretirse bu sayı küçük çıkar ve ayırıcı bunu kullanarak sahte batch'i kolayca tanır. Mini-batch ayrımının parametresiz hâlidir.
- **Eşitlenmiş öğrenme oranı.** Tüm ağırlıklar $\mathcal N(0,1)$'den başlatılır ve He ölçeği başlatmada değil, çalışma anında uygulanır: $\hat w_i=w_i\cdot c$, $c=\sqrt{2/\text{fan\_in}}$.
- **Piksel normalizasyon katmanı** (üreticide her konvolüsyondan sonra). Her piksel konumunda kanal vektörü birim RMS'e ölçeklenir: $b_{x,y}=a_{x,y}\big/\sqrt{\frac1C\sum_ja_{x,y}^{j\,2}+\epsilon}$. Bu, üretici ile ayırıcı arasındaki rekabet yüzünden büyüklüklerin kontrolden çıkmasını önler (BN'nin yerine).

:::türetim Eşitlenmiş öğrenme oranı neyi eşitler?
Adam ve RMSProp gibi uyarlamalı optimizasyon yöntemleri adımı gradyanın ölçeğine böler. Bu yüzden bir parametrenin adımı kabaca $\eta$ büyüklüğündedir ve ölçekten bağımsızdır.
- **Doğrudan He başlatması**: $w\sim\mathcal N(0,c^2)$. Adım $\approx\eta$, ağırlığın tipik büyüklüğü $c$; göreli değişim $\eta/c=\eta\sqrt{\text{fan\_in}/2}$. Girdisi çok olan katmanlar (küçük $c$) göreli olarak çok daha hızlı değişir.
- **Çalışma anında ölçekleme**: saklanan $w\sim\mathcal N(0,1)$, etkin ağırlık $\hat w=cw$. Adım $w$'de $\approx\eta$, etkin ağırlıkta $\approx c\eta$; göreli değişim $c\eta/c=\eta$. **Her katmanda aynıdır.**

Başlangıçtaki dağılım iki durumda aynıdır ($\hat w\sim\mathcal N(0,c^2)$); fark yalnız optimizasyon dinamiğindedir. Ek x3 bu iki oranı ölçüyor.
:::

Tüm bu tekniklerin birleşimi yazarların son derece inandırıcı, yüksek çözünürlüklü yüz görüntüleri üretmesini sağladı. Ama "inandırıcı" tam olarak ne demek? Değerlendirme, GAN'larla çalışırken büyük zorluklardan biridir. Üretilen görüntülerin çeşitliliğini otomatik olarak değerlendirmek mümkün olsa da kalitelerini yargılamak çok daha zor ve öznel bir iştir. İnsan değerlendiriciler kullanmak bir yoldur, ama pahalı ve zaman alıcıdır. Bu yüzden yazarlar, üretilen görüntülerin yerel görüntü yapısını eğitim görüntülerinkiyle **her ölçekte** karşılaştırmayı önerdi.

:::uyarı Kaynak notu: "çok ölçekli benzerlik" ve StyleGAN
ProGAN makalesindeki ölçüt, Laplace piramidinin her düzeyinden alınan 7 × 7'lik yamaların dağılımlarını dilimlenmiş Wasserstein uzaklığıyla (sliced Wasserstein distance, SWD) karşılaştırır. Slayt 93, bu fikrin StyleGAN'a "yol açtığını" söyler; bu, kaynak kitabın anlatı köprüsüdür. StyleGAN makalesinin kendi gerekçesi stil aktarımı literatürüdür (AdaIN, Huang ve Belongie 2017) ve değerlendirmede SWD değil FID ile algısal yol uzunluğu kullanılır.
:::

### StyleGAN {#stylegan}

Yüksek çözünürlüklü görüntü üretiminde en iyi sonuç, aynı Nvidia ekibinin 2018'deki (CVPR 2019) bir makalesiyle yeniden ileri taşındı; makale popüler **StyleGAN** mimarisini tanıttı. Yazarlar, üretilen görüntülerin her ölçekte eğitim görüntüleriyle aynı yerel yapıya sahip olmasını sağlamak için üreticide stil aktarımı tekniklerini kullandı ve üretilen görüntülerin kalitesini büyük ölçüde iyileştirdi. Ayırıcı ve kayıp fonksiyonu değiştirilmedi, yalnız üretici değişti. StyleGAN üreticisi iki ağdan oluşur:

**Eşleme ağı.** Gizil temsilleri $z$ (yani kodları) bir $w$ vektörüne eşleyen sekiz katmanlı bir MLP. Bu vektör sonra birden çok afin dönüşümden (aktivasyonsuz yoğun katmanlar; şekildeki "A" kutuları) geçirilir ve birden çok vektör üretilir. Bu vektörler üretilen görüntünün stilini, ince dokulardan (ör. saç rengi) üst düzey özelliklere (ör. yetişkin ya da çocuk) kadar farklı düzeylerde kontrol eder. Kısacası eşleme ağı kodları birden çok stil vektörüne eşler.

**Sentez ağı.** Görüntüleri üretmekten sorumludur. Sabit, öğrenilen bir girdisi vardır. (Açık olmak gerekirse bu girdi eğitimden sonra sabittir, ama eğitim sırasında geri yayılımla sürekli ayarlanır.) Bu girdiyi daha önceki gibi birden çok konvolüsyon ve büyütme katmanından geçirir, ama iki farkla:
1. Girdiye ve konvolüsyon katmanlarının tüm çıktılarına (aktivasyon fonksiyonundan önce) biraz **gürültü** eklenir.
2. Her gürültü katmanını bir **uyarlamalı örnek normalizasyonu** (AdaIN) katmanı izler. AdaIN her öznitelik haritasını bağımsız olarak standartlaştırır (öznitelik haritasının ortalamasını çıkarıp standart sapmasına bölerek). Sonra stil vektörünü kullanarak her öznitelik haritasının ölçeğini ve kaymasını belirler; stil vektörü her öznitelik haritası için bir ölçek ve bir bias terimi içerir:
$$\mathrm{AdaIN}(\mathbf x_i,\mathbf y)=y_{s,i}\,\frac{\mathbf x_i-\mu(\mathbf x_i)}{\sigma(\mathbf x_i)}+y_{b,i}.$$

<figure class="fig wide"><svg viewBox="0 0 640 340" role="img" aria-label="StyleGAN üreticisi: z, piksel normalizasyonu ve 8 tam bağlı katmanla w'ye eşlenir; sentez ağı sabit bir 4×4 girdiden başlar, her düzeyde gürültü eklenir ve w'den türetilen stil AdaIN ile uygulanır"><defs><marker id="s1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="s2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cp-f"/></marker><marker id="s3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><rect x="20" y="20" width="80" height="28" rx="4" class="soft-b"/><text x="60.0" y="38.0" text-anchor="middle" class="sm">z ~ N(0, I)</text><rect x="20" y="70" width="80" height="28" rx="4" class="soft-b"/><text x="60.0" y="88.0" text-anchor="middle" class="sm">piksel normu</text><rect x="20" y="118" width="80" height="11" rx="2" class="soft-b"/><rect x="20" y="132" width="80" height="11" rx="2" class="soft-b"/><rect x="20" y="146" width="80" height="11" rx="2" class="soft-b"/><rect x="20" y="160" width="80" height="11" rx="2" class="soft-b"/><rect x="20" y="174" width="80" height="11" rx="2" class="soft-b"/><rect x="20" y="188" width="80" height="11" rx="2" class="soft-b"/><rect x="20" y="202" width="80" height="11" rx="2" class="soft-b"/><rect x="20" y="216" width="80" height="11" rx="2" class="soft-b"/><text x="60" y="176" text-anchor="middle" class="sm">8 × (FC</text><text x="60" y="190" text-anchor="middle" class="sm">+ LReLU)</text><rect x="20" y="258" width="80" height="28" rx="4" class="soft-p"/><text x="60.0" y="276.0" text-anchor="middle" class="sm">w</text><path d="M60,48 L60,68" class="ink" marker-end="url(#s1)"/><path d="M60,98 L60,116" class="ink" marker-end="url(#s1)"/><path d="M60,232 L60,256" class="ink" marker-end="url(#s1)"/><text x="60" y="305" text-anchor="middle" class="tb">eşleme ağı</text><rect x="200" y="20" width="120" height="28" rx="4" class="soft-a"/><text x="260.0" y="38.0" text-anchor="middle" class="sm">sabit 4 × 4 × 512</text><rect x="200" y="70" width="120" height="26" rx="4" class="soft-p"/><text x="260.0" y="87.0" text-anchor="middle" class="sm">gürültü + AdaIN</text><path d="M260,48 L260,68" class="ink" marker-end="url(#s1)"/><rect x="200" y="110" width="120" height="26" rx="4" class="soft-a"/><text x="260.0" y="127.0" text-anchor="middle" class="sm">conv 3 × 3</text><path d="M260,96 L260,108" class="ink" marker-end="url(#s1)"/><rect x="200" y="150" width="120" height="26" rx="4" class="soft-p"/><text x="260.0" y="167.0" text-anchor="middle" class="sm">gürültü + AdaIN</text><path d="M260,136 L260,148" class="ink" marker-end="url(#s1)"/><rect x="200" y="200" width="120" height="26" rx="4" class="soft-a"/><text x="260.0" y="217.0" text-anchor="middle" class="sm">2× büyüt, conv 3 × 3</text><path d="M260,176 L260,198" class="ink" marker-end="url(#s1)"/><rect x="200" y="240" width="120" height="26" rx="4" class="soft-p"/><text x="260.0" y="257.0" text-anchor="middle" class="sm">gürültü + AdaIN</text><path d="M260,226 L260,238" class="ink" marker-end="url(#s1)"/><rect x="200" y="285" width="120" height="26" rx="4" class="soft-a"/><text x="260.0" y="302.0" text-anchor="middle" class="sm">…  1024 × 1024</text><path d="M260,266 L260,283" class="ink" marker-end="url(#s1)"/><rect x="150" y="73" width="22" height="20" rx="3" class="soft-p"/><text x="161" y="87" text-anchor="middle" class="tp">A</text><path d="M172,83 L198,83" class="cp" marker-end="url(#s2)"/><rect x="348" y="73" width="22" height="20" rx="3" class="soft-b"/><text x="359" y="87" text-anchor="middle" class="tb">B</text><path d="M348,83 L322,83" class="cb" marker-end="url(#s3)"/><rect x="150" y="153" width="22" height="20" rx="3" class="soft-p"/><text x="161" y="167" text-anchor="middle" class="tp">A</text><path d="M172,163 L198,163" class="cp" marker-end="url(#s2)"/><rect x="348" y="153" width="22" height="20" rx="3" class="soft-b"/><text x="359" y="167" text-anchor="middle" class="tb">B</text><path d="M348,163 L322,163" class="cb" marker-end="url(#s3)"/><rect x="150" y="243" width="22" height="20" rx="3" class="soft-p"/><text x="161" y="257" text-anchor="middle" class="tp">A</text><path d="M172,253 L198,253" class="cp" marker-end="url(#s2)"/><rect x="348" y="243" width="22" height="20" rx="3" class="soft-b"/><text x="359" y="257" text-anchor="middle" class="tb">B</text><path d="M348,253 L322,253" class="cb" marker-end="url(#s3)"/><path d="M100,272 L130,272 L130,83 L150,83 M130,163 L150,163 M130,253 L150,253" class="cp"/><text x="260" y="330" text-anchor="middle" class="ta">sentez ağı</text><text x="378" y="20" class="sm">A: w → kanal başına (y<tspan baseline-shift="-3" font-size="75%">s</tspan>, y<tspan baseline-shift="-3" font-size="75%">b</tspan>) (öğrenilen afin)</text><text x="378" y="38" class="sm">AdaIN: y<tspan baseline-shift="-3" font-size="75%">s</tspan> · (x − μ(x)) / σ(x) + y<tspan baseline-shift="-3" font-size="75%">b</tspan></text><text x="378" y="56" class="sm">B: tek kanallı Gauss gürültüsü,</text><text x="378" y="70" class="sm">öğrenilen kanal ölçekleriyle eklenir</text><text x="378" y="110" class="tq">Stil karıştırma</text><text x="378" y="128" class="sm">ilk K düzey w₁, kalanlar w₂ (K rastgele)</text><text x="378" y="148" class="sm">kaba (4²–8²): poz, yüz biçimi</text><text x="378" y="164" class="sm">orta (16²–32²): yüz ayrıntıları, saç biçimi</text><text x="378" y="180" class="sm">ince (64²–1024²): renk, mikro yapı</text><text x="378" y="220" class="tq">Kırpma</text><text x="378" y="238" class="sm">w′ = w<tspan dx="-0.53em" dy="-0.2em">¯</tspan><tspan dx="0.17em" dy="0.2em"> + ψ (w − w<tspan dx="-0.53em" dy="-0.2em">¯</tspan><tspan dx="0.17em" dy="0.2em">), w<tspan dx="-0.53em" dy="-0.2em">¯</tspan><tspan dx="0.17em" dy="0.2em">: w’lerin ortalaması</tspan></tspan></tspan></text><text x="378" y="254" class="sm">ψ &lt; 1: kalite artar, çeşitlilik azalır</text></svg><figcaption>Şekil 3 — StyleGAN üreticisi (Karras, Laine ve Aila 2019). Gizil kod girdi katmanına değil, her düzeydeki AdaIN'e stil olarak girer. Ayırıcı ve kayıp ProGAN'dakiyle aynıdır, yalnız üretici değişti. Düzeylerin etkisi makaledeki stil karıştırma deneyinden.</figcaption></figure>

:::örnek El hesabı: AdaIN
Bir kanalın değerleri 1, 2, 3, 4; stil $y_s=2$, $y_b=1$. $\mu=2.5$, popülasyon varyansı $\frac14(2.25+0.25+0.25+2.25)=1.25$, $\sigma=1.118$. 4'ün çıktısı: $2\cdot\frac{4-2.5}{1.118}+1=2\cdot1.342+1=\mathbf{3.683}$. Normalizasyon, önceki katmanın stilini (kanalın ortalama ve ölçeğini) siler; yeni stil onun yerine yazılır. Bu yüzden her düzeydeki stil yalnız o düzeyden bir sonraki AdaIN'e kadar etkilidir: stiller **yerelleşir**.
:::

**Gürültü neden ayrı?** Kodlardan bağımsız gürültü eklemek fikri çok önemlidir. Bir görüntünün bazı kısımları oldukça rastgeledir, örneğin her çilin ya da saç telinin tam konumu. Önceki GAN'larda bu rastgelelik ya kodlardan gelmek ya da üreticinin kendisinin ürettiği sözde rastgele bir gürültü olmak zorundaydı.
- Kodlardan geliyorsa üretici kodların temsil gücünün önemli bir kısmını gürültü saklamaya ayırmak zorundaydı; bu oldukça israftır. Üstelik gürültü ağın içinden akıp üreticinin son katmanlarına ulaşabilmeliydi; bu, eğitimi muhtemelen yavaşlatan gereksiz bir kısıttır. Son olarak aynı gürültü farklı düzeylerde kullanıldığı için bazı görsel kusurlar ortaya çıkabilir.
- Üretici kendi sözde rastgele gürültüsünü üretmeye çalışırsa bu gürültü pek inandırıcı görünmeyebilir ve daha çok görsel kusura yol açar. Ayrıca üreticinin ağırlıklarının bir kısmı yine sözde rastgele gürültü üretmeye ayrılır; bu da israftır.

Ek gürültü girdileri eklenerek tüm bu sorunlardan kaçınılır: GAN, verilen gürültüyü kullanarak görüntünün her kısmına doğru miktarda rastgelelik ekleyebilir. Eklenen gürültü her düzey için farklıdır. Her gürültü girdisi Gauss gürültüsüyle dolu tek bir öznitelik haritasıdır; verilen düzeyin tüm öznitelik haritalarına yayınlanır ve eklenmeden önce öğrenilen öznitelik başına ölçek çarpanlarıyla ölçeklenir ("B" kutuları).

**Stil karıştırma.** Son olarak StyleGAN, **karıştırma düzenlileştirmesi** (ya da stil karıştırma) denen bir teknik kullanır: üretilen görüntülerin bir yüzdesi (makalede %90) iki farklı kodla üretilir. $\mathbf c_1$ ve $\mathbf c_2$ kodları eşleme ağından geçirilerek iki stil vektörü $\mathbf w_1$ ve $\mathbf w_2$ elde edilir. Sonra sentez ağı ilk $K$ düzey için $\mathbf w_1$, kalan düzeyler için $\mathbf w_2$ stiline dayalı bir görüntü üretir. Kesme düzeyi $K$ rastgele seçilir. Bu, ağın komşu düzeylerdeki stillerin ilişkili olduğunu varsaymasını önler; bu da GAN'da yerelliği teşvik eder, yani her stil vektörü üretilen görüntüdeki yalnız sınırlı sayıda özelliği etkiler.

:::sezgi Eşleme ağı neden var?
$z$ sabit bir dağılımdan (Gauss) gelmek zorundadır. Eğitim verisinde bazı özellik birleşimleri hiç yoksa (ör. uzun saçlı çocuk azsa), $z$'den görüntüye eşleme bu boşluğu kapatmak için bükülmek zorundadır: $z$'deki düz bir çizgi görüntüde birden çok özelliği aynı anda değiştirir (dolaşıklık). $w$ ise hiçbir sabit dağılıma uymak zorunda değildir; eşleme ağı bu bükülmeyi üstlenir ve $w$ uzayı daha **ayrık** (disentangled) olabilir. Makale bunu algısal yol uzunluğu ve doğrusal ayrılabilirlik ölçütleriyle ölçer: $w$ uzayında ikisi de $z$ uzayından iyidir.
:::

**Kırpma hilesi** (truncation trick). Örnek alırken $w$, ortalamasına doğru çekilir: $\mathbf w'=\bar{\mathbf w}+\psi(\mathbf w-\bar{\mathbf w})$, $\psi<1$. Gizil uzayın seyrek (düşük yoğunluklu) bölgelerinden gelen kodlar üretici tarafından iyi öğrenilmemiştir; onları ortalamaya çekmek kaliteyi artırır ama çeşitliliği azaltır. El hesabı: $\bar w=0$, $\psi=0.7$, $w=2$ ise $w'=\mathbf{1.4}$. (Ad BigGAN'dan, Brock ve ark. 2018; orada $z$'nin eşiği aşan bileşenleri yeniden örneklenir. StyleGAN'da kırpma $w$ uzayında yapılır ve yalnız kaba düzeylere uygulanabilir.)

Aşağıdaki deneyde kusurlu bir üreticiyi kırparak kalite (kesinlik) ile çeşitliliğin (geri çağırma) nasıl değiş tokuş edildiğini ve modların nasıl düştüğünü gör.

::lab l2

**StyleGAN2: AdaIN'den demodülasyona.** StyleGAN görüntülerinde damla biçimli (droplet) kusurlar görülür. Karras ve ark. (2020) bunu AdaIN'in her öznitelik haritasını **ayrı ayrı** normalize etmesine bağlar: üretici, göreli büyüklük bilgisini normalizasyondan kaçırmak için tek bir noktada çok büyük bir tepe oluşturur. Çözüm, normalizasyonu veriden değil **ağırlıklardan** yapmaktır. Stil konvolüsyon ağırlığını modüle eder, $w'_{ijk}=s_i\,w_{ijk}$ (giriş kanalı $i$, çıkış kanalı $j$, çekirdek konumu $k$). Girdiler birim varyanslı ve bağımsızsa çıkış $j$'nin standart sapması $\sqrt{\sum_{i,k}w'^2_{ijk}}$'dir; demodülasyon bu sayıya böler:
$$w''_{ijk}=w'_{ijk}\Big/\sqrt{\textstyle\sum_{i,k}w'^2_{ijk}+\epsilon}.$$
Böylece çıkış ölçeği beklentide 1 olur, ama gerçek aktivasyonlar normalize edilmediği için kanallar arası göreli büyüklükler korunur. Aşama e bunu örnek başına ağırlıklarla grup konvolüsyonu olarak kuruyor.

Çok çeşitli GAN'lar vardır; hepsini kapsamak bütün bir kitap gerektirir. Bu bölüm ana fikirleri vermeyi amaçlıyor. Yalnız hızla etkileyici sonuçlar almak istiyorsanız önceden eğitilmiş bir model kullanabilirsiniz (ör. StyleGAN'ın resmî ağırlıkları). Slayt 103–104'teki ünlü yüzleri ve nesne görüntüleri metne dökülemedi.

### PyTorch'ta {#pytorch}

| kavram | PyTorch | not |
|---|---|---|
| DCGAN üretici bloğu | `nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()` | kenarı 2 katına çıkarır |
| DCGAN ayırıcı bloğu | `nn.Conv2d(c_in, c_out, 4, 2, 1), nn.LeakyReLU(0.2)` | ilk blokta BN yok |
| DCGAN optimizasyonu | `Adam(lr=2e-4, betas=(0.5, 0.999))` | makaledeki değerler |
| doymayan kayıp | `F.binary_cross_entropy_with_logits(D(G(z)), ones)` | M5.2 |
| FID | `torchmetrics.image.fid.FrechetInceptionDistance(feature=2048)` | büyük N (≥ 10 000) |
| IS | `torchmetrics.image.inception.InceptionScore()` | |
| AdaIN | `F.instance_norm(x) * ys[:, :, None, None] + yb[:, :, None, None]` | |
| piksel normu | `x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-8)` | |
| mini-batch std | `torch.cat([x, x.std(0).mean().expand(B, 1, H, W)], 1)` | |
| eşitlenmiş öğrenme oranı | `F.conv2d(x, self.weight * self.c, ...)`, `self.c = (2 / fan_in) ** 0.5` | ağırlık N(0, 1) |
| kırpma | `w = w_avg + psi * (w - w_avg)` | `w_avg`: eğitimde EMA |

### Özet {#ozet}

- GAN'ın tek dengesi ($p_g=p_{data}$, $D=\frac12$) yalnız sonsuz kapasite ve fonksiyon uzayında geçerlidir; ulaşılması garanti değildir.
- Mod çöküşü: üretici bir moda yığılır, ayırıcı diğerlerini unutur, üretici modlar arasında döner. Çareler: deneyim tekrarı, mini-batch ayrımı, öznitelik eşleme, TTUR, doymayan kayıp.
- DCGAN: havuz yerine adımlı/transposed konvolüsyon, BN (üretici çıkışı ve ayırıcı girişi hariç), tam bağlı gizli katman yok, üreticide ReLU + tanh, ayırıcıda LeakyReLU. (4, 2, 1) bloğu kenarı ikiye katlar (4 → 8).
- GAN olabilirlik vermez. IS $=\exp I(x;y)=\exp(H(y)-\mathbb EH(y\mid x))\in[1,C]$; gerçek veriye bakmaz, kandırılabilir (el örneği 1.445). FID $=\|\Delta\mu\|^2+\mathrm{tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$; 1-B'de $(\Delta\mu)^2+(\Delta\sigma)^2$ (el örneği 2). FID yanlıdır ve $N$'ye bağlıdır.
- ProGAN: 4 × 4'ten 1024 × 1024'e büyüme, yumuşak geçiş (α = 0.3: 2.6), mini-batch std, eşitlenmiş öğrenme oranı (göreli adımı her katmanda eşitler), piksel normu.
- StyleGAN: eşleme ağı (z → w, daha ayrık uzay), sabit girdi, gürültü enjeksiyonu (stokastik ayrıntı), AdaIN (el örneği 3.683), stil karıştırma (yerellik), kırpma (kalite ↔ çeşitlilik; 1.4). StyleGAN2: AdaIN yerine ağırlık demodülasyonu.
