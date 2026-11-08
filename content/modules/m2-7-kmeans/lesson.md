### Problem: etiket yokken yapı bulmak {#problem}

Şimdiye kadarki her yöntem **denetimliydi**: $(x_i,y_i)$ çiftleri vardı. **Kümeleme**de yalnız $x_i$'ler var; amaç benzer örnekleri gruplamak. Kullanım: müşteri segmentasyonu, görüntü sıkıştırma (renk paletini $k$ renge indirmek — vektör nicemleme), belge gruplama, bir modelin gömme uzayında yapı keşfi, büyük veri kümelerinden temsilci örnek seçmek.

"Benzer" ve "grup" tanımsız kavramlardır; her kümeleme yöntemi kendi tanımını getirir. **k-means**'in tanımı en basitidir: her kümenin bir **merkezi** vardır ve iyi bir kümeleme, noktaların kendi merkezlerine **kare uzaklıklarının toplamını** küçük tutar.

### Amaç: inertia {#amac}

$k$ merkez $c_1,\dots,c_k$ ve her noktanın küme ataması $z_i\in\{1..k\}$:
$$J(C,z)=\sum_{i=1}^n\|x_i-c_{z_i}\|^2.$$
**Inertia** (küme içi kare toplamı). Bu, bir sıkıştırma hatası olarak da okunabilir: her noktayı merkeziyle değiştirirsek (vektör nicemleme) yapılan kare hata.

$J$'yi küresel olarak en küçüklemek NP-zordur (düzlemde bile, Mahajan ve ark. 2009). Lloyd algoritması bunun yerine iki kolay alt problemi sırayla çözer.

### Lloyd algoritması {#lloyd}

:::türetim İki adım, her biri J'yi en küçükler
**Ata (C sabit):** $J=\sum_i\|x_i-c_{z_i}\|^2$; her terim yalnız kendi $z_i$'sine bağlı, ayrı ayrı küçültülür: $z_i=\arg\min_k\|x_i-c_k\|^2$ (en yakın merkez).
**Güncelle (z sabit):** $J=\sum_k\sum_{i:z_i=k}\|x_i-c_k\|^2$; her $c_k$ ayrı bir kuadratik. Türev $-2\sum_{i:z_i=k}(x_i-c_k)=0\Rightarrow c_k=\frac1{n_k}\sum_{i:z_i=k}x_i$ (kümenin ortalaması; M0.4'te ortalamanın kare kaybı en küçükleyen sabit olduğunu görmüştük).
**Sonlu sürede durur:** her adım $J$'yi artırmaz; olası atamaların sayısı sonlu ($k^n$) ve aynı atamaya geri dönmek $J$'nin kesin azalmasıyla çelişir (eşitlik kuralı sabitse). Bu yüzden algoritma sonlu adımda durur — ama bir **yerel** minimumda (ne atama ne de merkezler tek başına değiştirilerek iyileştirilemeyen bir nokta).
:::

**Vektörize hâli:** atama, M2.1'deki mesafe matrisinin satır argmin'idir; güncelleme, one-hot atama matrisi $A\in\{0,1\}^{n\times k}$ ile $C=\mathrm{diag}(A^\top\mathbf1)^{-1}A^\top X$ (modülün **c** aşaması). **Boş küme:** bir merkeze hiç nokta düşmezse ortalaması tanımsızdır; seçenekler: merkezi değiştirmemek (modülümüzde), en uzak noktaya taşımak, yeniden rastgele başlatmak.

<figure class="fig wide"><svg viewBox="0 0 640 200" role="img" aria-label="Lloyd algoritmasının iki adımı: noktaları en yakın merkeze ata, merkezleri kümelerin ortalamasına taşı"><g class="sm"><g transform="translate(10,10)"><text x="90" y="12" text-anchor="middle">1. başlangıç merkezleri</text><g class="ink-f"><circle cx="30" cy="60" r="4"/><circle cx="45" cy="80" r="4"/><circle cx="25" cy="95" r="4"/><circle cx="55" cy="55" r="4"/><circle cx="130" cy="120" r="4"/><circle cx="150" cy="140" r="4"/><circle cx="140" cy="160" r="4"/><circle cx="160" cy="115" r="4"/></g><rect x="70" y="60" width="12" height="12" class="ca-f"/><rect x="95" y="100" width="12" height="12" class="cb-f"/></g><g transform="translate(220,10)"><text x="90" y="12" text-anchor="middle">2. ata: en yakın merkez</text><g class="ca-f"><circle cx="30" cy="60" r="4"/><circle cx="45" cy="80" r="4"/><circle cx="25" cy="95" r="4"/><circle cx="55" cy="55" r="4"/></g><g class="cb-f"><circle cx="130" cy="120" r="4"/><circle cx="150" cy="140" r="4"/><circle cx="140" cy="160" r="4"/><circle cx="160" cy="115" r="4"/></g><rect x="70" y="60" width="12" height="12" class="ca-f"/><rect x="95" y="100" width="12" height="12" class="cb-f"/><line x1="10" y1="135" x2="170" y2="35" class="mute dash"/></g><g transform="translate(430,10)"><text x="90" y="12" text-anchor="middle">3. güncelle: ortalamaya taşı</text><g class="ca-f"><circle cx="30" cy="60" r="4"/><circle cx="45" cy="80" r="4"/><circle cx="25" cy="95" r="4"/><circle cx="55" cy="55" r="4"/></g><g class="cb-f"><circle cx="130" cy="120" r="4"/><circle cx="150" cy="140" r="4"/><circle cx="140" cy="160" r="4"/><circle cx="160" cy="115" r="4"/></g><rect x="33.75" y="66.5" width="12" height="12" class="ca-f"/><rect x="139" y="127" width="12" height="12" class="cb-f"/><line x1="76" y1="66" x2="40" y2="72" class="ca dash"/><line x1="101" y1="106" x2="145" y2="133" class="cb dash"/></g></g></svg><figcaption>Şekil 1 — Kareler merkezler, daireler noktalar. Atama adımı uzayı merkezlere göre Voronoi hücrelerine böler (kesikli çizgi); güncelleme adımı her merkezi kendi hücresindeki noktaların ortalamasına taşır. İki adım, değişiklik kalmayana kadar tekrarlanır.</figcaption></figure>

::lab l1

### Başlatma: k-means++ {#plusplus}

Lloyd yerel bir minimumda durduğu için sonuç başlatmaya bağlıdır. Rastgele $k$ nokta seçmek iki merkezi aynı kümeye, hiçbirini başka bir kümeye koyabilir; algoritma bunu genelde düzeltemez (iki merkez bir kümeyi paylaşır, iki küme bir merkezi).

**k-means++** (Arthur ve Vassilvitskii 2007): ilk merkez rastgele bir veri noktası; sonraki her merkez, noktanın **en yakın mevcut merkeze kare uzaklığı** $D(x)^2$ ile orantılı olasılıkla seçilir. Uzaktaki (henüz temsil edilmeyen) bölgeler daha olası seçilir. Kanıtlanmış garanti: beklenen inertia, en iyi çözümün en fazla $O(\log k)$ katıdır — üstüne Lloyd yalnız iyileştirir. Pratikte ayrıca **birden çok başlatma** (scikit-learn'de `n_init`) yapılıp en düşük $J$'li sonuç seçilir; modülün **e** aşamasında $R$ başlatmayı tek tensörde paralel koşturacaksın.

### Kaç küme? {#k}

$J$'nin en küçük değeri $k$ arttıkça **hep** azalır ($k=n$'de 0): $J$ tek başına $k$ seçemez.

- **Dirsek yöntemi:** $J(k)$ grafiğinde eğimin belirgin azaldığı nokta. Öznel; gerçek veride dirsek çoğu zaman belirsizdir.
- **Siluet:** her nokta için $a$ = kendi kümesindeki noktalara ortalama uzaklık, $b$ = en yakın başka kümeye ortalama uzaklık, $s=\frac{b-a}{\max(a,b)}\in[-1,1]$. Ortalama siluet en büyük olan $k$. Maliyeti $O(n^2)$.
- **Olasılıksal modeller:** GMM ile BIC/AIC (M2.8) — karmaşıklığı cezalandırılmış olabilirlik.
- **Görev:** kümeler bir sonraki görev için kullanılıyorsa (ör. sıkıştırma), $k$'yı o görevin doğrulama performansıyla seç.

### k-means neyi varsayar, ne zaman başarısız olur? {#varsayimlar}

Merkez + Öklid uzaklığı ile tanımlanan kümeler **Voronoi hücreleridir**: dışbükeydirler ve sınırları iki merkezin dik ortay düzlemidir. Örtük varsayımlar:

1. Kümeler **küresel** (her yönde aynı yayılım) — uzamış kümeler ortadan bölünür.
2. Kümeler kabaca **eşit boyutta ve yoğunlukta** — kalabalık, geniş bir kümeyi bölmek, küçük yoğun kümeleri birleştirmekten "ucuz"dur.
3. Kümeler **dışbükey** — ay, halka ya da sarmal biçimli kümeleri hiçbir $k$ yakalayamaz.
4. Öznitelikler aynı ölçekte (M2.1: ölçek uzaklığı belirler).

Aşağıda bu başarısızlıkları dirsek ve siluetle birlikte gör.

::lab l2

**Çözümler:** uzamış ve farklı boyutlu kümeler için her kümeye kendi kovaryansını veren **Gauss karışım modeli** (M2.8); dışbükey olmayan kümeler için yoğunluk tabanlı **DBSCAN** ya da benzerlik çizgesine dayanan **spektral kümeleme**; ölçek sorunları için standartlaştırma ya da öğrenilmiş bir gömme.

### Derin öğrenmede k-means {#dl}

- **Vektör nicemleme (VQ):** VQ-VAE ve ses/görüntü belirteçleyicileri sürekli gömmeleri bir k-means kod kitabına (codebook) eşler; kod kitabı ya k-means güncellemesiyle (üstel hareketli ortalama) ya da gradyanla öğrenilir (M5).
- **Öz-denetimli öğrenme:** DeepCluster ve SwAV, gömmeleri k-means (ya da benzeri) ile kümeleyip küme atamalarını sahte etiket olarak kullanır.
- **Model sıkıştırma:** ağırlıkları $k$ değere nicemlemek (ağırlık paylaşımı).
- **Veri seçimi:** büyük bir veri kümesinden çeşitli bir alt küme seçmek için gömmeleri kümeleyip her kümeden örnek almak.

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| amaç | inertia $J=\sum_i\|x_i-c_{z_i}\|^2$ |
| Lloyd | ata (en yakın merkez) ↔ güncelle (küme ortalaması); $J$ hiç artmaz, sonlu adımda yerel minimum |
| vektörize | atama = mesafe matrisi argmin; güncelleme $\mathrm{diag}(A^\top\mathbf1)^{-1}A^\top X$ |
| başlatma | k-means++ ($D^2$ ile örnekleme), birden çok başlatma |
| $k$ seçimi | $J$ hep düşer; dirsek, siluet, BIC, görev |
| varsayımlar | küresel, eşit boyutlu, dışbükey kümeler; ölçekli öznitelikler |

**Kaynaklar:** Lloyd (1982), "Least squares quantization in PCM"; Arthur ve Vassilvitskii (2007), "k-means++: the advantages of careful seeding"; Hastie ve ark., *ESL* §14.3; Bishop, *PRML* §9.1; Rousseeuw (1987), siluet; scikit-learn kullanıcı kılavuzu, "Clustering" (yöntem karşılaştırması); van den Oord ve ark. (2017), VQ-VAE; Caron ve ark. (2018), DeepCluster.
