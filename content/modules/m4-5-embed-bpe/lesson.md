### Kelimeleri sayılara çevirmek {#token}

Bilgisayarlar kelimeleri (ya da sembolleri) doğrudan işleyemez; onları temsil etmenin bir yolu gerekir. İki adımda yapılır: (1) **tokenleştirme** (tokenization) — metni bir sözlükteki birimlere (token) bölüp her birine bir tamsayı kimlik (ID) vermek; (2) her kimliği bir **vektöre** çevirmek. Birim seçimi bir ödünleşimdir:

| birim | sözlük boyu | dizi uzunluğu | bilinmeyen kelime |
|---|---|---|---|
| kelime | çok büyük (yüz binler) | kısa | sorun: eğitimde görülmemiş kelime |
| karakter | çok küçük (≈ 100) | çok uzun | yok |
| alt-kelime (BPE, WordPiece, Unigram) | orta (≈ 30–200 bin) | orta | yok (karakterlere/baytlara iner) |

**Sözlük dışı (OOV) kovaları.** Kategorik öznitelikler (ve kelimeler) özel bir işlem gerektirir. Kategori sayısını eğitim verisinden alıp buna göre eğitebilirsin; ama test ya da dağıtım sırasında **daha önce hiç görülmemiş** kategorilerle karşılaşabilirsin. Kelimelerde de aynı sorun vardır: çıkarımda eğitim verisinde olmayan kelimeler görülür. Çözüm **OOV kovalarıdır**: birkaç yeni (yer tutucu) kategori/kelime eklenir; eğitimde görülmemiş bir kategori/kelime görülünce (ör. kategorinin özetiyle) bir OOV kovasına atanır. Bu özellikle kategori sayısı büyük (posta kodları, şehirler, kelimeler, ürünler, kullanıcılar) ya da sürekli değişiyorsa iyi bir fikirdir: tam listeyi elde etmek kullanışlı ya da mümkün olmayabilir. Alt-kelime tokenleştirme OOV sorununu kökünden çözer.

### One-hot kodlama {#onehot}

**Örnek: California konut veri seti.** 1990 nüfus sayımına dayanan, bir California bölgesindeki evlere ait özet istatistikler: boylam, enlem, ortanca konut yaşı, toplam oda, toplam yatak odası, nüfus, hane sayısı, ortanca gelir, ortanca ev değeri ve **ocean_proximity** (okyanusa yakınlık) — sayısal **olmayan** tek öznitelik. Beş olası değeri vardır: `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`. Kelimeleri sinir ağına doğrudan veremeyiz; bu özniteliği önce kodlamamız gerekir.

Kategori çok az (5) olduğu için **one-hot** kodlama kullanılabilir: her kategoriyi bir dizine (0–4) eşle; $n$. sınıfın one-hot vektörü $n$. konumunda 1, diğer her yerde 0 olan $N$ boyutlu vektördür ($N$: sınıf sayısı + OOV kovaları).

:::türetim One-hot vektörlerin geometrisi
Farklı iki kategori $i\ne j$ için one-hot vektörler $\mathbf e_i,\mathbf e_j$:
$$\|\mathbf e_i-\mathbf e_j\|^2=\underbrace{1^2}_{i.\text{ konum}}+\underbrace{(-1)^2}_{j.\text{ konum}}=2,\qquad\|\mathbf e_i-\mathbf e_j\|=\sqrt2,\qquad\mathbf e_i^\top\mathbf e_j=0 .$$
**Her** çiftin uzaklığı aynı ($\sqrt2$), iç çarpımı aynı (0): one-hot kodlama "NEAR BAY", "NEAR OCEAN"a "INLAND"dan daha mı yakın? sorusuna hiçbir cevap vermez. Vektörler birbirine diktir (ortonormal); kategoriler arasındaki benzerlik bilgisi yoktur.
:::

:::uyarı Kaynak düzeltmesi
Ders slaytında "one-hot kodlamada iki sınıf arasındaki uzaklık her zaman 2'dir" deniyor. 2, uzaklığın **karesidir**; Öklid uzaklığı $\sqrt2\approx1.414$'tür. Slaytın asıl noktası doğru: uzaklık her çift için **aynıdır** ve iç çarpım her zaman sıfırdır.
:::

100 kategori olsaydı? 1 000? Milyonlarca? One-hot kodlama basittir ama en iyi çözüm değildir: her one-hot vektörün boyu sözlük uzunluğu artı OOV kova sayısıdır. Kategori azsa sorun yoktur, ama sözlük büyükse **embedding** ile kodlamak çok daha verimlidir. Kaba kural (Géron): 10'dan az kategori → one-hot; 50'den fazla → embedding; arasında ikisini de dene.

### Embedding: öğrenilen yoğun vektörler {#embedding}

**Embedding**, bir kategoriyi (kelimeyi, tokenı) temsil eden **eğitilebilir yoğun bir vektördür**. Varsayılan olarak rastgele başlatılır: örneğin "NEAR BAY" başlangıçta $[0.131, 0.890]$ gibi rastgele bir vektörle, "NEAR OCEAN" $[0.631, 0.791]$ gibi başka bir rastgele vektörle temsil edilebilir. Bu örnekte 2-B embedding kullanıyoruz; boyut ayarlanabilir bir hiperparametredir (kelime embedding'lerinde tipik olarak 100–1 000). Embedding'ler eğitilebilir olduğu için eğitim boyunca yavaş yavaş iyileşir: "NEAR BAY" ve "NEAR OCEAN" oldukça benzer kategoriler olduğundan gradyan inişi onları büyük olasılıkla birbirine **yaklaştırır**, "INLAND"ın embedding'inden de **uzaklaştırmaya** çalışır.

Gerçekten de temsil ne kadar iyiyse sinir ağının doğru tahmin yapması o kadar kolaydır; dolayısıyla eğitim embedding'leri kategorilerin **yararlı temsillerine** dönüştürme eğilimindedir. Buna **temsil öğrenmesi** (representation learning) denir. Embedding'ler genellikle yalnız eldeki görev için değil, başka görevler için de yeniden kullanılabilir. Girdiyi embedding'e çevirince girdi uzayımız embedding vektör uzayına (ör. $\mathbb R^{256}$) geçer; bu uzayda uzaklık (çoğu zaman **kosinüs benzerliği**, M0.1) benzerliği ölçer.

<figure class="fig wide"><svg viewBox="0 0 560 180" role="img" aria-label="One-hot vektör ile embedding matrisinin çarpımı, matrisin ilgili satırını seçer"><rect x="20" y="70" width="18" height="18" class="mute"/><text x="29" y="83" text-anchor="middle" class="sm">0</text><rect x="38" y="70" width="18" height="18" class="mute"/><text x="47" y="83" text-anchor="middle" class="sm">0</text><rect x="56" y="70" width="18" height="18" class="soft-b cq"/><text x="65" y="83" text-anchor="middle" class="sm">1</text><rect x="74" y="70" width="18" height="18" class="mute"/><text x="83" y="83" text-anchor="middle" class="sm">0</text><rect x="92" y="70" width="18" height="18" class="mute"/><text x="101" y="83" text-anchor="middle" class="sm">0</text><text x="20" y="62" class="sm">one-hot x (V = 5)</text><text x="126" y="84" text-anchor="middle">×</text><rect x="142" y="34" width="30" height="18" class="mute"/><text x="157" y="47" text-anchor="middle" class="sm">0.1</text><rect x="172" y="34" width="30" height="18" class="mute"/><text x="187" y="47" text-anchor="middle" class="sm">-0.3</text><rect x="202" y="34" width="30" height="18" class="mute"/><text x="217" y="47" text-anchor="middle" class="sm">0.8</text><rect x="142" y="52" width="30" height="18" class="mute"/><text x="157" y="65" text-anchor="middle" class="sm">0.5</text><rect x="172" y="52" width="30" height="18" class="mute"/><text x="187" y="65" text-anchor="middle" class="sm">0.2</text><rect x="202" y="52" width="30" height="18" class="mute"/><text x="217" y="65" text-anchor="middle" class="sm">-0.1</text><rect x="142" y="70" width="30" height="18" class="soft-p cp"/><text x="157" y="83" text-anchor="middle" class="sm">-0.7</text><rect x="172" y="70" width="30" height="18" class="soft-p cp"/><text x="187" y="83" text-anchor="middle" class="sm">0.9</text><rect x="202" y="70" width="30" height="18" class="soft-p cp"/><text x="217" y="83" text-anchor="middle" class="sm">0.4</text><rect x="142" y="88" width="30" height="18" class="mute"/><text x="157" y="101" text-anchor="middle" class="sm">0.3</text><rect x="172" y="88" width="30" height="18" class="mute"/><text x="187" y="101" text-anchor="middle" class="sm">0.3</text><rect x="202" y="88" width="30" height="18" class="mute"/><text x="217" y="101" text-anchor="middle" class="sm">0.3</text><rect x="142" y="106" width="30" height="18" class="mute"/><text x="157" y="119" text-anchor="middle" class="sm">0.0</text><rect x="172" y="106" width="30" height="18" class="mute"/><text x="187" y="119" text-anchor="middle" class="sm">-0.6</text><rect x="202" y="106" width="30" height="18" class="mute"/><text x="217" y="119" text-anchor="middle" class="sm">0.2</text><text x="142" y="26" class="sm">W (V × d = 5 × 3)</text><text x="248" y="84" text-anchor="middle">=</text><rect x="264" y="70" width="30" height="18" class="soft-p cp"/><text x="279" y="83" text-anchor="middle" class="sm">-0.7</text><rect x="294" y="70" width="30" height="18" class="soft-p cp"/><text x="309" y="83" text-anchor="middle" class="sm">0.9</text><rect x="324" y="70" width="30" height="18" class="soft-p cp"/><text x="339" y="83" text-anchor="middle" class="sm">0.4</text><text x="264" y="62" class="sm">W[2] (2. satır)</text><text x="20" y="150" class="sm">İleri: çarpım yapmaya gerek yok — dizinle satırı oku (lookup).</text><text x="20" y="168" class="sm">Geri: ∂L/∂W = xᵀ · ∂L/∂e → yalnız 2. satır gradyan alır; aynı token birden çok kez geçerse gradyanlar o satırda toplanır.</text></svg><figcaption>Şekil 1 — Embedding katmanı = one-hot × matris. One-hot girdinin ardından aktivasyonsuz ve bias'sız bir yoğun katman koymak, ağırlık matrisi embedding matrisi olan bir embedding katmanına denktir; ama lookup hem bellek hem hesap olarak çok daha ucuzdur (V × d'lik çarpım yerine d sayının kopyası).</figcaption></figure>

:::türetim Embedding katmanı = one-hot × matris; gradyanı = satırlara dağıtım
Sözlük boyu $V$, embedding boyu $d$, embedding matrisi $\mathbf W\in\mathbb R^{V\times d}$. Token $k$'nın one-hot vektörü $\mathbf x=\mathbf e_k$:
$$\mathbf x^\top\mathbf W=\sum_{i}x_i\mathbf W_{i,:}=\mathbf W_{k,:}.$$
Yani one-hot ardından aktivasyonsuz ve bias'sız bir yoğun katman, ağırlık matrisinin $k$. satırını **seçer**: bir embedding katmanıdır (ders slaytındaki örnek: 20 kategori, 10 boyut → 20 girdili one-hot'a bağlanan 10 nöronlu yoğun katman; eğitimden sonra 20 grup × 10 ağırlık, 20 embedding vektörüdür; `nn.Embedding(20, 10)` ile aynı). **Geri geçiş:** $\mathbf e=\mathbf x^\top\mathbf W$ için $\partial L/\partial\mathbf W=\mathbf x\,(\partial L/\partial\mathbf e)^\top$ — $k$. satır dışındaki tüm satırların gradyanı sıfırdır. Bir batch'te aynı token birden çok kez geçerse, her geçişin gradyanı **aynı satıra toplanır** (scatter-add; PyTorch'ta `index_add_`, M3.3'teki gradyan biriktirme). Bu yüzden `nn.Embedding(..., sparse=True)` seyrek gradyan üretebilir ve nadir tokenlar az güncelleme alır.
:::

PyTorch'ta `nn.Embedding(num_embeddings=V, embedding_dim=d, padding_idx=0)`: girdi tamsayı tensörü (her şekil), çıktı sonuna $d$ eklenmiş tensör. `padding_idx` satırı sıfırda sabitlenir ve gradyan almaz (M4.4'teki dolgu tokenları için).

Aşağıdaki deneyde ocean_proximity'nin 2-B embedding'ini eğit. Hedef iki sayı (fiyat ve kıyıya yakınlık); model $\hat{\mathbf y}=\mathbf W\mathbf e_c+\mathbf b$. Rastgele başlayan vektörlerin eğitimle nasıl düzenlendiğini, benzer kategorilerin yaklaştığını ve **nadir** bir kategorinin (ISLAND) neden yerinden pek kımıldamadığını izle.

::lab l1

### Kelime embedding'leri ve word2vec {#word2vec}

En yaygın örnek **kelime embedding'leridir**: bir doğal dil işleme görevinde çalışırken kendi kelime embedding'lerini eğitmek yerine önceden eğitilmiş olanları yeniden kullanmak çoğu zaman daha iyidir. Kelimeleri vektörlerle temsil etme fikri 1960'lara uzanır ve sinir ağları dahil birçok gelişmiş teknik kullanıldı. Belki de en önemli adım Tomáš Mikolov ve ark.'nın 2013 tarihli iki makalesidir ("Efficient Estimation of Word Representations in Vector Space" ve "Distributed Representations of Words and Phrases and their Compositionality"): sinir ağlarıyla kelime embedding'i öğrenmenin, önceki girişimleri belirgin biçimde geride bırakan **verimli** bir yolunu buldular — **word2vec**. Bu verimlilik, çok büyük bir metin derlemi üzerinde, herhangi bir kelimenin yakınındaki kelimeleri tahmin eden bir ağ eğitmeyi mümkün kıldı.

Sonuç yalnız yakınlık değildir: embedding'ler uzayda **anlamlı eksenler** boyunca da düzenlenir. İki ünlü örnek:
$$\text{King}-\text{Man}+\text{Woman}\approx\text{Queen},\qquad\text{Madrid}-\text{Spain}+\text{France}\approx\text{Paris}.$$
İkincisinde "başkenti" ilişkisine karşılık gelen bir vektör bileşeni vardır: ülke–başkent çiftleri arasındaki fark vektörleri, PCA ile 2-B'ye indirilmiş bir çizimde kabaca **paralel** yatay oklardır (word2vec boyutları 100–1 000 aralığındadır; çizim için boyut indirgenir).

word2vec iki modelle eğitilebilir: **CBOW** (continuous bag of words) — yakındaki kelimelerden ortadaki kelimeyi tahmin et; **skip-gram** — ortadaki kelimeden yakındaki kelimeleri tahmin et. Ayrıntılar, negatif örnekleme ve PMI matris ayrışımı bağlantısı M4.12'de. Önceden eğitilmiş embedding'ler: Google'ın **word2vec**'i, Stanford'un **GloVe**'u, Facebook'un **FastText**'i (alt-kelime karakter n-gramlarıyla OOV'ye dayanıklı). Bu embedding katmanları bir kelimeyi öğrenilmiş vektörüne bir arama tablosu gibi eşler; büyük veri kümelerinde dikkatle seçilmiş yardımcı NLP görevleriyle eğitildikleri için oldukça iyidirler (M3.9'daki öz-gözetimli ön eğitimin erken örneği).

### Alt-kelime tokenleştirme ve BPE {#bpe}

Kelime düzeyinde embedding'ler birçok dil için uygun değildir — özellikle Türkçe gibi eklemeli dillerde: "evlerimizden" sözlükte hiç görülmemiş olabilir ama "ev", "ler", "imiz", "den" çok sıktır. Kelimeler harflerden oluşur: **karakter düzeyinde** embedding'ler görülmemiş kelimelerin de embedding'ini üretebilir — OOV sorunu çözülür. Bazı görevlerde başarılı oldukları gösterildi ama popüler değiller; sorun **hesap yüküdür**: 5 kelimelik bir cümle 30–40 embedding vektörü üretebilir (ve dikkat mekanizmasının maliyeti uzunluğun karesiyle büyür, M4.6).

**Bayt çifti kodlaması** (byte pair encoding, BPE) popüler bir alt-kelime tokenleştirme yöntemidir; BPE ile tokenleştirip ortaya çıkan tokenlar için embedding öğrenmek bugün başarılı yaklaşımdır. NLP'ye Sennrich, Haddow ve Birch ("Neural Machine Translation of Rare Words with Subword Units", ACL 2016) uyarladı. BPE aslında bir **sıkıştırma** algoritmasıdır (Gage 1994): en sık geçen bayt çiftini bul ve sözlüğe yeni bir "bayt" olarak ekle.

:::tanım BPE eğitimi ve kodlaması
**Eğitim:** her kelimeyi karakterlerine böl ve sonuna bir kelime sonu işareti (`</w>`) ekle. Sonra $K$ kez tekrarla: derlemdeki tüm **komşu sembol çiftlerini** (kelime sıklığıyla ağırlıklı) say; en sık çifti $(a,b)$ seç; her yerde $a\,b\to ab$ birleştir; $(a,b)$'yi birleştirme listesine ekle. Sözlük = başlangıç karakterleri + $K$ yeni sembol.

**Kodlama** (yeni bir kelime): karakterlerine böl, `</w>` ekle, sonra öğrenilen birleştirmeleri **öğrenildikleri sırayla** uygula. Görülmemiş bir kelime bile en kötü durumda karakterlerine iner: OOV yoktur.
:::

:::örnek Sennrich ve ark.'nın örneği
Derlem (sıklıklar): low 5, lower 2, newest 6, widest 3. Başlangıçta komşu çiftler: (e, s) 6 + 3 = **9** (newest, widest), (s, t) 9, (t, `</w>`) 9, (w, e) 6 + 2 = 8, (l, o) 7, (o, w) 7, … Eşitlikte ilk görüleni seçersek birleştirmeler: 1) e + s → es (9), 2) es + t → est (9), 3) est + `</w>` → est`</w>` (9), 4) l + o → lo (7), 5) lo + w → low (7), 6) n + e (6), 7) ne + w (6), 8) new + est`</w>` (6), 9) low + `</w>` (5), 10) w + i (3). 10 birleştirmeden sonra: low`</w>` · low e r `</w>` · newest`</w>` · wi d est`</w>`. Görülmemiş "lowest" → low · est`</w>`: iki tanıdık parça. `est</w>` (kelime sonundaki "est") ile ortadaki "est" farklı tokenlardır — kelime sonu işareti ekleri ayırt eder.
:::

Aşağıdaki deneyde birleştirmeleri tek tek uygula, çift sıklık tablosunun nasıl değiştiğini izle, kendi derlemini yaz (her satır: kelime sıklık) ve öğrenilen birleştirmelerle yeni kelimeler kodla. "Türkçe ekler" derleminde BPE'nin kökleri ve ekleri ("ev", "kitap", "lar", "de␣") kendiliğinden bulduğuna bak.

::lab l2

:::ayrıntı Bayt düzeyi BPE, WordPiece, Unigram ve SentencePiece
- **Bayt düzeyi BPE** (GPT-2): başlangıç sembolleri karakterler değil 256 **bayttır** (UTF-8). Her metin baytlardan oluştuğu için hiçbir şey sözlük dışı kalmaz; emoji ve yabancı alfabeler de tokenleşir. GPT-2'nin sözlüğü ≈ 50 257 token.
- **WordPiece** (BERT): BPE'ye benzer ama birleştireceği çifti sıklığa göre değil, birleşmenin derlemin olabilirliğini en çok artırmasına göre seçer (≈ $\text{sayı}(ab)/(\text{sayı}(a)\,\text{sayı}(b))$); kelime içi parçalar `##` ile işaretlenir.
- **Unigram** dil modeli (Kudo 2018): büyük bir aday sözlükle başlar, olabilirliğe en az katkı veren tokenları budar; bir kelimenin birden çok tokenleştirmesini olasılıklarla değerlendirir.
- **SentencePiece**: boşluğu da sıradan bir karakter (`▁`) sayarak dil bağımsız tokenleştirme yapan kütüphane (BPE ya da Unigram); T5, LLaMA gibi modellerde.
:::

### Sinir ağı bigram modeli = sayım modeli {#bigram}

Embedding'lerin ne öğrendiğini en basit dil modelinde görmek öğreticidir. **Bigram modeli**: bir sonraki tokenın olasılığı yalnız bir öncekine bağlıdır, $P(w_t\mid w_{t-1})$. İki yol:
1. **Sayım:** $\hat P(b\mid a)=\dfrac{\text{sayı}(a,b)}{\sum_{b'}\text{sayı}(a,b')}$ — en çok olabilirlik kestirimi (M0.4).
2. **Sinir ağı:** önceki tokenın embedding'i → doğrusal katman → softmax; çapraz entropiyle eğit. En basit hâlinde $\mathbf W\in\mathbb R^{V\times V}$, $\text{logit}=\mathbf W_{a,:}$ (embedding boyu = $V$).

:::türetim İkisi aynı noktaya yakınsar
$a$'yı izleyen tokenların ortalama çapraz entropisi $L_a=-\sum_bn_{ab}\log\mathrm{softmax}(\mathbf z_a)_b$ ($n_{ab}$: sayı). M1.5'teki softmax gradyanıyla $\partial L_a/\partial z_{ab}=n_a\,p_b-n_{ab}$ ($n_a=\sum_bn_{ab}$). Sıfıra eşitle: $p_b=n_{ab}/n_a$ — sayım modelinin MLE'si. $\mathbf W$'nin her satırı bağımsız bir softmax'tır ve kayıp dışbükeydir (M1.5); dolayısıyla gradyan inişi sayım modeline yakınsar (hiç görülmemiş çiftlerde logit $-\infty$'a gider; düzenlileştirme ya da sayımlara $+1$ eklemek — Laplace düzeltmesi, M2.2 — ikisinde de bunu önler). Embedding boyu $V$'den küçükse ($d\ll V$) ağ artık her çifti ezberleyemez: benzer bağlamlarda geçen tokenlar benzer embedding'ler almak zorundadır — **genelleme** buradan gelir (Bengio ve ark. 2003). Modülün PyTorch aşamasında bunu ölçüyorsun.
:::

### Bağlamsal embedding'ler {#baglamsal}

Şimdiye kadar öğrenilen embedding'lerin **sabit** olduğunu varsaydık. Sorun: bir kelimenin (ya da alt-kelime tokenının) anlamı **bağlama** bağlıdır — birçok anlamı olan kelimeler bunu açıkça gösterir ("yüz": sayı, surat, yüzmek; "bank": banka, sıra). Bağlamsal kelime embedding'i fikri ChatGPT dahil modern derin dil modellerinin temelidir: modele başlangıç embedding'lerini verirsin, model **dikkat** mekanizmasıyla bu embedding'leri **dönüştürür**, öyle ki dönüştürülmüş embedding'ler o cümledeki kelimeleri daha iyi temsil eder. Dikkati M4.6'da, Transformer'ı M4.7'de, bağlamsal embedding'leri ve önceden eğitilmiş dil modellerini M4.16'da görüyoruz.

### Özet {#ozet}

- Metin → token kimlikleri → vektörler. Kelime/karakter/alt-kelime ödünleşimi; OOV kovaları.
- One-hot: her çift arası $\sqrt2$, iç çarpım 0 — benzerlik bilgisi yok; büyük sözlükte verimsiz (10'dan az kategori one-hot, 50'den fazla embedding).
- Embedding: eğitilebilir yoğun vektör; one-hot × $\mathbf W$ = satır seçimi; gradyan yalnız kullanılan satırlara toplanır (scatter-add) — nadir tokenlar az öğrenir.
- word2vec (CBOW, skip-gram): anlamlı eksenler, analojiler; önceden eğitilmiş word2vec/GloVe/FastText.
- BPE: en sık komşu çifti birleştir, birleştirmeleri sırayla uygula; OOV yok; bayt düzeyi BPE, WordPiece, Unigram, SentencePiece.
- Sinir ağı bigram modeli sayım MLE'sine yakınsar; $d\ll V$ embedding'ler genelleme sağlar. Sabit embedding'lerin sınırı: bağlam — dikkat ve bağlamsal embedding'ler.
