### Fikir: benzer girdiler, benzer çıktılar {#fikir}

Şimdiye kadarki modeller (M1) veriden bir **parametre vektörü** öğrendi ve eğitimden sonra veriyi bir kenara bıraktı. **k-en yakın komşu (k-NN)** tam tersini yapar: hiçbir parametre öğrenmez, eğitim verisini **saklar**. Yeni bir $x$ için:

1. $x$ ile her eğitim örneği arasındaki uzaklığı hesapla.
2. En yakın $k$ örneği bul.
3. Sınıflandırmada çoğunluk oyunu, regresyonda ortalamayı döndür.

Arkasındaki varsayım **pürüzsüzlük**tür: girdi uzayında yakın noktaların çıktıları da benzerdir. Bu tür yöntemlere **parametrik olmayan** (nonparametric) ya da **örnek tabanlı** (instance-based) denir: modelin karmaşıklığı veriyle birlikte büyür.

| | parametrik (lojistik regresyon) | k-NN |
|---|---|---|
| eğitim | optimizasyon, pahalı | yok (veriyi sakla), $O(1)$ |
| tahmin | $O(d)$ | $O(nd)$ — tüm veriye uzaklık |
| bellek | $O(d)$ | $O(nd)$ |
| karar sınırı | doğrusal | keyfi karmaşık |
| varsayım | log-odds doğrusal | yakın = benzer |

<figure class="fig"><svg viewBox="0 0 420 230" role="img" aria-label="Sorgu noktası çevresinde k = 5 en yakın komşuyu içeren çember ve çoğunluk oyu"><g class="sm"><circle cx="210" cy="115" r="62" class="mute dash" fill="none"/><circle cx="210" cy="115" r="5" class="ink-f"/><text x="216" y="111">?</text><g class="ca-f"><circle cx="180" cy="90" r="5"/><circle cx="245" cy="140" r="5"/><circle cx="230" cy="70" r="5"/><circle cx="90" cy="60" r="5"/><circle cx="70" cy="140" r="5"/><circle cx="120" cy="190" r="5"/></g><g class="cb-f"><circle cx="190" cy="160" r="5"/><circle cx="330" cy="70" r="5"/><circle cx="350" cy="160" r="5"/><circle cx="300" cy="200" r="5"/><circle cx="380" cy="110" r="5"/></g><g class="cp-f"><circle cx="160" cy="125" r="5"/><circle cx="40" cy="30" r="5"/><circle cx="380" cy="30" r="5"/></g><text x="275" y="40" class="ta">k = 5 içinde: 3 mavi,</text><text x="275" y="55" class="ta">1 turuncu, 1 yeşil → mavi</text><line x1="210" y1="115" x2="272" y2="115" class="ink"/><text x="236" y="129">r₅</text></g></svg><figcaption>Şekil 1 — Sorgu noktasına (siyah) en yakın 5 örnek kesikli çemberin içindedir; çoğunluk oyu sınıfı belirler. Çemberin yarıçapı veriye ve k'ya göre değişir: yoğun bölgelerde küçük, seyrek bölgelerde büyük.</figcaption></figure>

### Uzaklık ölçüleri {#uzaklik}

"Yakın"ın anlamı uzaklık ölçüsüne bağlıdır (M0.1):

| ölçü | formül | ne zaman |
|---|---|---|
| Öklid (L2) | $\sqrt{\sum_j(a_j-b_j)^2}$ | varsayılan, sürekli öznitelikler |
| Manhattan (L1) | $\sum_j\lvert a_j-b_j\rvert$ | aykırı değerlere daha az duyarlı, ızgara benzeri uzaylar |
| Chebyshev (L∞) | $\max_j\lvert a_j-b_j\rvert$ | en kötü öznitelik farkı önemliyse |
| kosinüs uzaklığı | $1-\frac{a\cdot b}{\|a\|\|b\|}$ | metin, gömme vektörleri (büyüklük değil yön önemli) |
| Hamming | farklı koordinat sayısı | ikili/kategorik öznitelikler |

:::uyarı Öznitelik ölçeği uzaklığı belirler
Bir öznitelik metre, diğeri kilometre cinsindense Öklid uzaklığı neredeyse yalnız metre özniteliğinin farkıdır; diğeri yok sayılır. Ölçek bir modelleme kararıdır ama çoğu zaman bilinçsizce verilir. Varsayılan: her özniteliği standartlaştır (ortalama 0, std 1) — eğitim verisinin istatistikleriyle, doğrulama/test'e aynı dönüşümü uygulayarak (M1.3 sızıntı uyarısı).
:::

### k'nın rolü: yanlılık–varyans {#k}

- **k = 1:** her eğitim noktası kendi bölgesine sahiptir (**Voronoi** hücreleri). Eğitim hatası her zaman 0'dır; tek bir gürültülü etiket çevresinde yanlış bir "ada" oluşturur. Düşük yanlılık, yüksek varyans.
- **k büyük:** oylama daha çok noktayı ortalar; sınırlar yumuşar, gürültü bastırılır. $k=n$'de her sorgu çoğunluk sınıfını alır. Yüksek yanlılık, düşük varyans.

Eğitim hatası $k$'yı seçmek için **kullanılamaz** (k = 1 hep kazanır). Seçenekler: ayrı bir doğrulama kümesi, $k$-katlı çapraz doğrulama ya da k-NN'ye özgü ucuz bir yöntem: **birini dışarıda bırak** (LOO) — her eğitim noktasını, kendisi hariç komşularıyla tahmin et. k-NN'de LOO bedavadır: yeniden eğitim yok, yalnız kendi uzaklığını yok say.

Pratik ayrıntılar: ikili sınıflandırmada **tek** $k$ beraberliği önler; çok sınıfta beraberlik yine olabilir, kuralı açıkça belirle (bu modülde: eşit oyda küçük etiket; eşit uzaklıkta küçük indeks). **Ağırlıklı oy** ($1/\text{uzaklık}$) yakın komşulara daha çok söz hakkı verir ve $k$'ya duyarlılığı azaltır.

::lab l1

:::ayrıntı Kuram: 1-NN, Bayes hatasının en fazla iki katı
**Bayes hatası** $R^*$, gerçek dağılımı bilen en iyi sınıflandırıcının hatasıdır (indirgenemez gürültü). Cover ve Hart (1967): $n\to\infty$ iken 1-NN'nin hatası $R_{1NN}$ için $R^*\le R_{1NN}\le2R^*(1-R^*)\le2R^*$ (iki sınıfta). Sezgi: çok veri olunca en yakın komşu sorguyla neredeyse aynı noktadadır; ikisinin etiketi bağımsız olarak aynı koşullu dağılımdan çekilir, uyuşmama olasılığı $2p(1-p)\le2\min(p,1-p)$. Ayrıca $k\to\infty$ ve $k/n\to0$ iken k-NN **tutarlıdır**: hatası $R^*$'a yakınsar (Stone 1977). **Ne demez:** sonlu veride, özellikle yüksek boyutta, ne kadar hızlı yakınsadığını — aşağıya bak.
:::

### Boyutun laneti {#lanet}

k-NN'nin varsayımı "yakın komşular gerçekten yakındır"dır. Yüksek boyutta bu varsayım çöker:

1. **Uzaklıklar yoğunlaşır.** $d$ boyutlu birim küpte düzgün dağılmış noktalar için iki nokta arasındaki kare uzaklık $d$ tane bağımsız terimin toplamıdır: ortalaması $d/6$, standart sapması $\sim\sqrt d$. Göreli yayılım $\sim\sqrt d/d=1/\sqrt d\to0$. En yakın ve en uzak komşu neredeyse aynı uzaklıktadır (Beyer ve ark. 1999). M0.1'deki "yüksek boyutta rastgele vektörler neredeyse diktir" deneyinin uzaklık versiyonu.
2. **Komşuluklar yerel değildir.** Verinin $r$ oranını içeren alt küpün kenarı $r^{1/d}$'dir. $d=10$'da %1 veri için kenarın %63'ü, $d=100$'de %95'i gerekir: "yakın komşular" uzayın her yerinden gelir.
3. **Gereken veri üstel büyür:** her eksende 10 hücrelik bir çözünürlük için $10^d$ hücre.

::lab l2

**Neden yine de işe yarar?** Gerçek veri (görüntüler, sesler, metin gömmeleri) yüksek boyutlu uzayın küçük, düşük boyutlu bir yapısına (**manifold**) yakındır; etkili boyut düşüktür. Ayrıca iyi bir **temsil** (bir sinir ağının öğrendiği gömme) anlamsal olarak benzer örnekleri birbirine yaklaştırır. Ham pikseller üzerinde k-NN CIFAR-10'da ~%35 doğruluk verirken (CS231n), iyi bir öz-denetimli gömme üzerinde k-NN %90'ı aşabilir — fark uzaklığın hesaplandığı uzaydadır.

### Vektörizasyon: mesafe matrisi {#vektorizasyon}

$n_{te}$ sorgu ve $n_{tr}$ eğitim noktası için tüm kare uzaklıklar $D\in\mathbb R^{n_{te}\times n_{tr}}$. Üç uygulama, aynı sonuç:

1. **İki döngü:** her $(i,j)$ çifti için $\|x_i-x_j\|^2$. $n_{te}n_{tr}$ Python iterasyonu: çok yavaş.
2. **Tek döngü:** her sorgu için `((Xtr - x) ** 2).sum(1)` — yayın, $(n_{tr},d)$'lik bir ara tensör.
3. **Döngüsüz:** kareyi aç.

:::türetim ‖a − b‖² = ‖a‖² + ‖b‖² − 2a·b ile tek matris çarpımı
$\|a-b\|^2=(a-b)\cdot(a-b)=a\cdot a-2a\cdot b+b\cdot b$. Tüm çiftler için: $s_{te}=$ her sorgunun kare normu $(n_{te},1)$, $s_{tr}=$ her eğitim noktasınınki $(1,n_{tr})$ ve $G=X_{te}X_{tr}^\top$ $(n_{te}\times n_{tr})$ tüm iç çarpımlar:
$$D=s_{te}+s_{tr}-2G\qquad(\text{yayınla }(n_{te}\times n_{tr})).$$
Maliyet bir matris çarpımıdır ($2n_{te}n_{tr}d$ FLOP, M0.2) ve BLAS ile çok hızlıdır. Ara bellek $(n_{te}\times n_{tr})$ — $(n_{te},n_{tr},d)$'lik yayın tensöründen $d$ kat küçük.
:::

:::uyarı Sayısal yok etme
Açılım, iki büyük sayının farkını alır: $a\approx b$ ise $\|a\|^2+\|b\|^2$ ile $2a\cdot b$ neredeyse eşittir ve fark yuvarlama hatasına boğulur. Sonuç küçük **negatif** sayılar olabilir (karekök alınca `nan`). Çare: `clamp_min(0)`; yüksek hassasiyet gerekiyorsa verileri önce ortala (normları küçültür) ya da float64 kullan. `torch.cdist` bu yüzden küçük boyutlarda doğrudan farkı, büyüklerde açılımı kullanır (`compute_mode` argümanı).
:::

**En küçük k'yı seçmek:** tam sıralama $O(n\log n)$; yalnız en küçük $k$ gerekiyorsa `torch.topk(D, k, largest=False)` (kısmi seçim). Çok sayıda $k$'yı birden değerlendirmek için sıralamayı bir kez yap, oyları **kümülatif** topla (modülün **e** aşaması).

**Bellek:** $n_{te}=10^4$, $n_{tr}=10^6$ float32 $D$ = 40 GB. Sorguları parçalar hâlinde işle ve her parçada yalnız `topk` sonucunu sakla.

### Hızlı arama {#arama}

Her sorgu için tüm veriye uzaklık ($O(nd)$) büyük $n$'de pahalıdır. İki aile:

- **Kesin arama yapıları:** KD-ağacı (uzayı eksen hizalı düzlemlerle böler), top ağacı (iç içe küreler). Düşük boyutta ($d\lesssim20$) sorgu $O(\log n)$'e yaklaşır; yüksek boyutta (laneti hatırla) budama çalışmaz ve kaba aramaya döner.
- **Yaklaşık en yakın komşu (ANN):** kesinlikten biraz vazgeçip çok hız kazanır. Yerellik duyarlı karma (LSH, rastgele izdüşümlerle — M0.1'deki işaret bölgeleri), çizge tabanlı yöntemler (HNSW), nicemleme (ürün nicemleme). FAISS, ScaNN gibi kütüphaneler milyarlarca vektörde milisaniyelerle arama yapar. Gömme tabanlı bilgi erişimi (retrieval-augmented generation), öneri sistemleri ve görsel arama bunlara dayanır.

### Derin öğrenmede k-NN {#dl}

- **Temsil değerlendirme:** öz-denetimli bir modelin gömmelerinin kalitesini ölçmenin standart yolu, donmuş gömmeler üzerinde k-NN doğruluğudur (eğitim gerektirmez).
- **Az örnekli öğrenme:** prototipik ağlar, her sınıfın ortalama gömmesine en yakın sınıfı seçer — k-NN'nin sınıf merkezli bir biçimi.
- **Bellek destekli modeller:** kNN-LM, dil modelinin tahminini eğitim verisindeki en yakın bağlamların sonraki kelimeleriyle karıştırır.
- **Veri temizliği:** her örneğin komşularıyla etiket uyuşmazlığı, yanlış etiketli örnekleri bulmanın basit bir yoludur.

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| algoritma | uzaklık → en yakın $k$ → oy (ya da ortalama); eğitim yok |
| $k$ | küçük: varyans; büyük: yanlılık; LOO ya da doğrulama ile seç |
| ölçek | öznitelikleri standartlaştır; uzaklık ölçüsü bir modelleme kararıdır |
| kuram | 1-NN ≤ 2 × Bayes hatası (sonsuz veride) |
| boyutun laneti | uzaklıklar yoğunlaşır, komşuluklar yerel değildir; iyi temsil şart |
| vektörizasyon | $D=s_{te}+s_{tr}-2X_{te}X_{tr}^\top$, `clamp_min(0)`, `topk` |

**Kaynaklar:** CS231n, "Image classification: data-driven approach, k-nearest neighbor" ve Assignment 1 (mesafe matrisinin üç sürümü); Cover ve Hart (1967), "Nearest neighbor pattern classification"; Hastie, Tibshirani, Friedman, *ESL* §2.5 ve §13.3 (boyutun laneti, k-NN); Beyer ve ark. (1999), "When is nearest neighbor meaningful?"; Johnson, Douze, Jégou (2019), "Billion-scale similarity search with GPUs" (FAISS); Malkov ve Yashunin (2018), HNSW; PyTorch `torch.cdist`, `torch.topk` belgeleri.
