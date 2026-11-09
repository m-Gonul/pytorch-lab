### Nereden geliyoruz: one-hot, OOV kovaları, embedding {#temel}

M4.5 kelimeyi bir tamsayı kimliğe, kimliği bir vektöre çevirmeyi anlattı. Bu modül, o vektörlerin **nasıl öğrenildiğini** ayrıntısıyla ele alıyor: word2vec'in iki modeli, negatif örnekleme, gürültü dağılımı, alt-örnekleme, bunların sayım tabanlı yöntemlerle (PMI, GloVe) ilişkisi ve alt-kelime vektörleri (fastText). Önce ders slaytlarının ilk yarısını kısaca toparlayıp birkaç noktayı derinleştirelim.

**Neden temsil gerekir?** Bir sinir ağı sayılarla çalışır; "INLAND" ya da "kraliçe" gibi bir sembolü doğrudan işleyemez. Slaytlardaki örnek California konut verisidir (1990 nüfus sayımına dayanan, bölge başına özet istatistikler). Dokuz özniteliğin hepsi sayısaldır; tek istisna `ocean_proximity`'dir ve beş değer alır: `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`. Bu sütun ağa verilmeden önce kodlanmalıdır.

**One-hot kodlama.** Her kategoriye 0'dan $N-1$'e bir dizin verilir. $n$. kategori, yalnız $n$. konumu 1 olan $N$ boyutlu $\mathbf e_n$ vektörüyle temsil edilir. Burada $N$ = sözlükteki kategori sayısı + OOV kovası sayısıdır. Bu vektörlerin hepsi birbirine diktir: $\mathbf e_i^\top\mathbf e_j=0$ ve $\|\mathbf e_i-\mathbf e_j\|=\sqrt2$ (M4.5'teki türetim ve kaynak düzeltmesi: slayttaki "uzaklık her zaman 2" ifadesindeki 2, uzaklığın **karesidir**). Sonuç olarak one-hot kodlama "NEAR BAY", "NEAR OCEAN"a "INLAND"dan daha mı yakın? sorusuna cevap veremez: her çift eşit uzaklıktadır.

**Sözlük dışı (OOV) kovaları.** Eğitimde görülen kategori listesi dağıtımda eksik kalır: yeni posta kodları, yeni ürünler, yeni kullanıcılar, yeni kelimeler gelir. Çözüm, sözlüğe birkaç **yer tutucu** kategori (kova) eklemektir. Görülmemiş bir değer, karma (hash) fonksiyonuyla bu kovalardan birine düşer. Kategori sayısı büyükse ya da sürekli değişiyorsa bu yöntem özellikle işe yarar, çünkü tam listeyi tutmak zor ya da imkânsızdır. Bedeli **çakışmadır**: farklı iki yeni değer aynı kovaya düşerse aynı vektörü paylaşır.

:::türetim Kaç çakışma beklenir?
$b$ kova ve eğitimde görülmemiş $m$ farklı değer olsun. Karmanın değerleri kovalara bağımsız ve düzgün dağıttığını **varsayalım**. Belirli iki değerin aynı kovaya düşme olasılığı $1/b$'dir. Beklentinin doğrusallığından, çakışan değer **çiftlerinin** beklenen sayısı
$$\mathbb E[\text{çakışan çift}]=\binom m2\frac1b=\frac{m(m-1)}{2b}$$
olur. Örnek: $b=100$, $m=20$ için $190/100=1.9$ çift. Hiç çakışma olmama olasılığı (doğum günü problemi) $\prod_{i=0}^{m-1}(1-i/b)$'dir; $m=20$, $b=100$'de ≈ 0.13. Yani küçük kova sayısında çakışma kuraldır, istisna değil. Kova sayısını artırmak çakışmayı $1/b$ ile azaltır ama her kovanın kendi parametreleri vardır ve nadir görülen kovalar az eğitilir.
:::

**Embedding.** 100, 1 000 ya da milyonlarca kategori olunca one-hot vektörler aynı ölçüde büyür ve hiçbir benzerlik bilgisi taşımaz. Géron'un kaba kuralı şöyledir: 10'dan az kategoride one-hot, 50'den fazlasında embedding, arada ikisini de dene. Embedding, her kategoriye karşılık gelen **eğitilebilir yoğun bir vektördür**. Başlangıçta rastgeledir; slayttaki örnekte "NEAR BAY" $[0.131, 0.890]$, "NEAR OCEAN" $[0.631, 0.791]$ gibi 2-B vektörlerle başlar. Eğitim boyunca gradyan inişi, görev açısından benzer davranan kategorileri yaklaştırır ("NEAR BAY" ile "NEAR OCEAN") ve farklı olanlardan ("INLAND") uzaklaştırır. Buna **temsil öğrenmesi** (representation learning) denir. Bu yolla öğrenilen vektörler çoğu zaman başka görevlerde de yeniden kullanılabilir.

:::türetim One-hot + biassız yoğun katman = embedding katmanı (slayttaki 20 × 10 örneği)
20 kategori, 10 boyutlu embedding. One-hot girdi $\mathbf x=\mathbf e_k\in\mathbb R^{20}$, 10 nöronlu, **aktivasyonsuz ve bias'sız** yoğun katman $\mathbf h=\mathbf W^\top\mathbf x$, $\mathbf W\in\mathbb R^{20\times10}$:
$$h_j=\sum_{i=1}^{20}W_{ij}x_i=W_{kj}\quad\Rightarrow\quad\mathbf h=\mathbf W_{k,:}.$$
Her one-hot girdinin tek 1'i 10 nörona 10 ağırlıkla bağlanır. 20 kategori için 20 grup × 10 ağırlık = **200 parametre** vardır; her grup bir kategorinin embedding vektörüdür. Bu, `nn.Embedding(20, 10)` ile birebir aynı hesaptır (Keras'ta `input_dim=20, output_dim=10`). Farkı maliyettedir. Yoğun katman $20\times10$ çarpma yapar ve bunların 190'ı sıfırla çarpımdır. Embedding katmanı ise satırı **dizinle okur**: 10 sayının kopyası. $V=10^5$, $d=300$'de bu fark $3\times10^7$ çarpma ile 300 kopya arasındaki farktır. Aslında one-hot girdiyle **her** katman bir tablo okumasına denktir, çünkü çıktı yalnız $k$'ya bağlıdır: bias ve aktivasyonla tablo $\phi(\mathbf W_{k,:}+\mathbf b)$ olurdu. Slayttaki koşulların (aktivasyon yok, bias yok) işlevi, tablonun **doğrudan** $\mathbf W$ olmasını sağlamaktır. Böylece katmanın parametreleri embedding katmanının parametreleriyle birebir eşleşir. Bias aynı ifade gücüne gereksiz bir parametre ekler; aktivasyon ise tablonun değer aralığını kısıtlar (ör. ReLU negatifleri siler).
:::

:::uyarı Kaynak notu: embedding uzayında "benzerliği Öklid uzaklığı ölçer"
Slayt, embedding uzayında benzerliği Öklid uzaklığının ölçtüğünü söylüyor. Kelime vektörlerinde pratikte **kosinüs benzerliği** kullanılır: word2vec'in analoji testi, en yakın komşu araması ve GloVe/fastText değerlendirmeleri hep kosinüsledir. Bunun nedeni, vektör **normunun** anlamdan çok başka etkenlerle değişmesidir: kelime sıklığı ve kullanım tutarlılığı gibi (Schakel ve Wilson 2015). Birim uzunluğa normalize edilmiş vektörlerde ikisi aynı sıralamayı verir: $\|\hat{\mathbf a}-\hat{\mathbf b}\|^2=2-2\cos(\mathbf a,\mathbf b)$.
:::

### Dağılımsal hipotez ve kısa tarihçe {#dagilimsal}

word2vec'in dayandığı fikir 1950'lere uzanır: **dağılımsal hipotez**. Benzer bağlamlarda geçen kelimelerin anlamları da benzerdir (Harris 1954). Firth'ün (1957) sık alıntılanan ifadesinin özü de budur: bir kelimeyi yanında bulunduğu kelimelerden tanırsın. "kral" ve "kraliçe" aynı kelimelerle (taç, saray, taht) geçer; "kral" ve "makarna" geçmez. O hâlde bir kelimeyi, **hangi bağlamlarda geçtiğini özetleyen** bir vektörle temsil edebiliriz.

Slayt bu fikrin vektörlerle uygulanmasının 1960'lara uzandığını söylüyor. Bilgi erişimindeki vektör uzayı modeli (Salton'un SMART sistemi, 1960'lar) ve Osgood'un anlamsal ayrım ölçekleri (1957) bu dönemin örnekleridir. Sonraki adımlar şöyle sıralanır:

| yıl | yöntem | fikir |
|---|---|---|
| 1990 | LSA (Deerwester ve ark.) | terim–belge sayım matrisinin kesik SVD'si |
| 2003 | sinirsel dil modeli (Bengio ve ark.) | önceki $n$ kelimenin embedding'lerinden sonrakini softmax ile tahmin et; embedding'ler yan ürün |
| 2008 | Collobert ve Weston | gerçek pencereyi bozulmuş pencereden ayıran sıralama kaybı; çok görevli NLP |
| 2013 | word2vec (Mikolov ve ark.) | gizli katmanı kaldır, yalnız embedding'leri öğren; hiyerarşik softmax / negatif örnekleme; milyarlarca kelimede saatler |
| 2014 | GloVe (Pennington ve ark.) | log eş-geçme sayımlarını ağırlıklı en küçük kareyle çarpanlara ayır |
| 2016–17 | fastText (Bojanowski ve ark.) | kelime = karakter n-gram vektörlerinin toplamı |

Mikolov ve ark.'nın iki makalesi slaytlarda da geçiyor. Birincisi "Efficient Estimation of Word Representations in Vector Space" (arXiv Ocak 2013, ICLR 2013 çalıştayı): CBOW ve skip-gram mimarileri, analoji test seti. İkincisi "Distributed Representations of Words and Phrases and their Compositionality" (NeurIPS 2013): negatif örnekleme, alt-örnekleme, ifade (phrase) öğrenme. Asıl yenilik **verimlilikti**. 2003 tarzı dil modelinin pahalı gizli katmanı ve tam softmax'ı kaldırılınca, birinci makalede tek makinede bir günden kısa sürede 1.6 milyar kelimeyle eğitim mümkün oldu.

:::uyarı Kaynak düzeltmesi: yazar adı
Slaytta yazarın adı iki yerde "Milokov", bir yerde "Tomaš Mikolov" olarak geçiyor. Doğrusu **Tomáš Mikolov**dur (Çekçe á).
:::

### Skip-gram: olasılık modeli {#skipgram}

Slayttaki tanım kısadır: skip-gram, **merkez** kelimeden yakınındaki kelimeleri tahmin eder; CBOW yakındaki kelimelerden merkezi tahmin eder. Önce skip-gram'ı tam olarak kuralım.

**Eğitim örnekleri.** Metin $w_1,\dots,w_T$, pencere $C$ olsun. Her $t$ konumu ve $1\le|j|\le C$ için $(w_t, w_{t+j})$ çifti bir eğitim örneğidir: $w_t$ **merkez** ($c$), $w_{t+j}$ **bağlam** ($o$, "output") kelimesidir. El hesabı: 5 kelimelik cümlede, pencere 2 için konumlara göre 2, 3, 4, 3, 2 komşu düşer; toplam **14** çift.

<figure class="fig wide"><svg viewBox="0 0 640 292" role="img" aria-label="Aynı cümle penceresinden skip-gram dört ayrı merkez-bağlam örneği, CBOW ise bağlam vektörlerinin ortalamasından tek bir merkez tahmini üretir"><defs><marker id="f1a" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><text x="20" y="16" class="sm">Cümle, pencere 2; merkez kelime: "sarayda"</text><rect x="12" y="20" width="56" height="22" rx="4" class="soft-a"/><text x="40" y="36" text-anchor="middle" class="sm">genç</text><rect x="74" y="20" width="56" height="22" rx="4" class="soft-a"/><text x="102" y="36" text-anchor="middle" class="sm">kral</text><rect x="136" y="20" width="56" height="22" rx="4" class="soft-b"/><text x="164" y="36" text-anchor="middle" class="sm">sarayda</text><rect x="198" y="20" width="56" height="22" rx="4" class="soft-a"/><text x="226" y="36" text-anchor="middle" class="sm">tahta</text><rect x="260" y="20" width="56" height="22" rx="4" class="soft-a"/><text x="288" y="36" text-anchor="middle" class="sm">oturdu</text><text x="20" y="80" class="tb">Skip-gram</text><text x="20" y="96" class="sm">merkez → her bağlam kelimesi ayrı bir örnek</text><rect x="40" y="110" width="70" height="26" rx="4" class="soft-b"/><text x="75" y="127" text-anchor="middle" class="sm">sarayda</text><text x="75" y="152" text-anchor="middle" class="sm">v = W_in[c]</text><line x1="112" y1="123" x2="196" y2="116" class="ink" marker-end="url(#f1a)"/><rect x="200" y="106" width="64" height="22" rx="4" class="soft-a"/><text x="232" y="121" text-anchor="middle" class="sm">genç</text><line x1="112" y1="123" x2="196" y2="146" class="ink" marker-end="url(#f1a)"/><rect x="200" y="136" width="64" height="22" rx="4" class="soft-a"/><text x="232" y="151" text-anchor="middle" class="sm">kral</text><line x1="112" y1="123" x2="196" y2="176" class="ink" marker-end="url(#f1a)"/><rect x="200" y="166" width="64" height="22" rx="4" class="soft-a"/><text x="232" y="181" text-anchor="middle" class="sm">tahta</text><line x1="112" y1="123" x2="196" y2="206" class="ink" marker-end="url(#f1a)"/><rect x="200" y="196" width="64" height="22" rx="4" class="soft-a"/><text x="232" y="211" text-anchor="middle" class="sm">oturdu</text><text x="20" y="246" class="sm">4 çift: (sarayda, genç), (sarayda, kral),</text><text x="20" y="262" class="sm">(sarayda, tahta), (sarayda, oturdu)</text><text x="20" y="282" class="sm">her biri: p(o | c) = softmax(W_out · v)[o]</text><text x="300" y="80" class="tp">CBOW</text><text x="300" y="96" class="sm">bağlam ortalaması → merkez (tek örnek)</text><rect x="300" y="106" width="64" height="22" rx="4" class="soft-a"/><text x="332" y="121" text-anchor="middle" class="sm">genç</text><line x1="366" y1="117" x2="410" y2="165" class="ink"/><rect x="300" y="136" width="64" height="22" rx="4" class="soft-a"/><text x="332" y="151" text-anchor="middle" class="sm">kral</text><line x1="366" y1="147" x2="410" y2="165" class="ink"/><rect x="300" y="166" width="64" height="22" rx="4" class="soft-a"/><text x="332" y="181" text-anchor="middle" class="sm">tahta</text><line x1="366" y1="177" x2="410" y2="165" class="ink"/><rect x="300" y="196" width="64" height="22" rx="4" class="soft-a"/><text x="332" y="211" text-anchor="middle" class="sm">oturdu</text><line x1="366" y1="207" x2="410" y2="165" class="ink"/><circle cx="420" cy="165" r="12" class="soft-p"/><text x="420" y="170" text-anchor="middle" class="sm">h</text><line x1="433" y1="165" x2="466" y2="165" class="ink" marker-end="url(#f1a)"/><rect x="470" y="154" width="70" height="22" rx="4" class="soft-b"/><text x="505" y="169" text-anchor="middle" class="sm">sarayda</text><text x="300" y="246" class="sm">h = (v_genç + v_kral + v_tahta + v_oturdu) / 4</text><text x="300" y="262" class="sm">1 örnek: p(c | bağlam) = softmax(W_out · h)[c]</text><text x="300" y="282" class="sm">sözcük sırası kaybolur ("torba")</text><rect x="560" y="104" width="22" height="70" class="soft-a"/><text x="571" y="190" text-anchor="middle" class="sm">W_in</text><text x="571" y="100" text-anchor="middle" class="sm">V×d</text><rect x="600" y="104" width="22" height="70" class="soft-p"/><text x="611" y="190" text-anchor="middle" class="sm">W_out</text><text x="611" y="100" text-anchor="middle" class="sm">V×d</text><text x="590" y="214" text-anchor="middle" class="sm">iki ayrı tablo</text></svg><figcaption>Şekil 1 — Skip-gram ve CBOW aynı pencereden farklı eğitim örnekleri çıkarır. Skip-gram her bağlam kelimesi için ayrı bir (merkez, bağlam) örneği üretir; CBOW bağlam kelimelerinin girdi vektörlerinin ortalamasıyla tek bir merkez tahmini yapar. İkisi de iki ayrı tablo öğrenir: girdi (W_in) ve çıktı (W_out) vektörleri; sonunda genellikle W_in gömme olarak kullanılır.</figcaption></figure>

**Model.** Her kelimenin **iki** vektörü vardır: merkezken kullanılan $\mathbf v_w$ (girdi tablosu $\mathbf W_{\text{in}}$'in satırı) ve bağlamken kullanılan $\mathbf u_w$ (çıktı tablosu $\mathbf W_{\text{out}}$'un satırı). Bağlam kelimesinin olasılığı softmax ile verilir:
$$p(o\mid c)=\frac{\exp(\mathbf u_o^\top\mathbf v_c)}{\sum_{w=1}^{V}\exp(\mathbf u_w^\top\mathbf v_c)}.$$
Amaç, derlemdeki ortalama log-olabilirliği en büyütmektir:
$$\frac1T\sum_{t=1}^{T}\ \sum_{-C\le j\le C,\ j\ne0}\log p(w_{t+j}\mid w_t).$$
Bu, M1.5'teki çok sınıflı lojistik regresyonun ta kendisidir. Girdi bir one-hot vektör ($\mathbf v_c=\mathbf W_{\text{in}}^\top\mathbf e_c$), "ağırlıklar" ise $\mathbf W_{\text{out}}$'tur. Ağda gizli katman ya da aktivasyon yoktur: tek doğrusal izdüşüm ve softmax.

:::sezgi Neden iki ayrı tablo?
Tek tablo kullanılsaydı ($\mathbf u_w=\mathbf v_w$), $p(\text{köpek}\mid\text{köpek})$ skoru $\mathbf v^\top\mathbf v=\|\mathbf v\|^2$ olurdu ve vektör büyüdükçe bu skor da büyürdü. Oysa bir kelime kendi bağlamında nadiren geçer, bu yüzden bu olasılık küçük olmalıdır. Tek tabloda ikisi çelişir (Goldberg ve Levy 2014'ün açıklaması). İki tablo bu kısıtı kaldırır. Eğitimden sonra genellikle $\mathbf W_{\text{in}}$ kullanılır; GloVe ise iki tablonun toplamını önerir (aşağıda).
:::

:::türetim Tam softmax'ın gradyanı ve maliyeti
Tek bir çift için kayıp $J=-\log p(o\mid c)=-\mathbf u_o^\top\mathbf v_c+\log\sum_w\exp(\mathbf u_w^\top\mathbf v_c)$. Log-toplam-üssün türevi softmax'tır (M1.5):
$$\frac{\partial J}{\partial\mathbf v_c}=-\mathbf u_o+\sum_{w}p(w\mid c)\,\mathbf u_w=\underbrace{\mathbb E_{w\sim p(\cdot\mid c)}[\mathbf u_w]}_{\text{modelin beklediği bağlam}}-\underbrace{\mathbf u_o}_{\text{gerçek bağlam}},\qquad
\frac{\partial J}{\partial\mathbf u_w}=\big(p(w\mid c)-\mathbb 1[w=o]\big)\,\mathbf v_c .$$
Gradyan, modelin beklediği bağlam vektörü ile gerçek bağlam vektörü arasındaki farktır. Model doğru tahmin ettiğinde ($p(o\mid c)\to1$) sıfıra gider. **Maliyet:** paydadaki toplam ve $\partial J/\partial\mathbf u_w$, **her** $w$ için hesaplanmalıdır. Çift başına $O(Vd)$ işlem gerekir. Örnek: $V=10^6$, $d=300$ için tek çiftte $3\times10^8$ çarpma-toplama (yalnız ileri skorlar). 1 milyar kelimelik bir derlemde kelime başına ortalama 6 çift (aşağıdaki dinamik pencere) varsa bir epoch $\approx1.8\times10^{18}$ işlem eder. Negatif örneklemeyle ($k=5$) çift başına $6\times300=1800$ işlem kalır: $V/(k+1)\approx1.7\times10^5$ kat daha az.
:::

**Dinamik pencere.** Uzak kelimeler merkezle daha az ilişkilidir. word2vec bunu ağırlıkla değil, **örneklemeyle** hesaba katar: her merkez kelime için etkin pencere $R$, $\{1,\dots,C\}$'den düzgün seçilir (Mikolov ve ark. 2013a, Bölüm 3.2; `word2vec.c`'de `b = next_random % window`).

:::türetim Uzaklığa göre örtük ağırlık
Uzaklığı $\delta$ olan bir komşu yalnız $R\ge\delta$ ise kullanılır:
$$P(R\ge\delta)=\frac{C-\delta+1}{C}.$$
$C=5$ için uzaklık 1, 2, 3, 4, 5'teki komşuların ağırlıkları $1,\ 0.8,\ 0.6,\ 0.4,\ 0.2$'dir. Cümle içindeki bir merkez kelimenin beklenen çift sayısı $2\,\mathbb E[R]=2\cdot\frac{C+1}2=C+1=6$'dır; sabit pencerede 10 olurdu. Böylece hem yakın komşular vurgulanır hem de eğitim ≈ %40 kısalır. GloVe aynı amaçla sayımları $1/\delta$ ile ağırlıklandırır.
:::

:::ayrıntı word2vec.c'de skip-gram "tersten" hesaplanır
Orijinal C kodunun skip-gram döngüsünde, merkez kelime $w_t$ çıktı tarafında (negatif örnekleme hedefi olarak) durur. Her bağlam kelimesinin **girdi** vektörü ise onu tahmin etmeye çalışır: kod $p(w_t\mid w_{t+j})$'yi modeller. Pencere simetrik olduğu için çift kümesi aynıdır: her $(a,b)$ çifti için $(b,a)$ da vardır. Bu yüzden derlem genelinde amaç değişmez. Fark, dinamik pencerenin hangi kelimeye göre çekildiği ve güncelleme sırasıdır. Bu ders sitesindeki kodlar (aşama c–d) makaledeki yönü kullanır.
:::

### CBOW: bağlamdan merkezi tahmin {#cbow}

**CBOW** (continuous bag-of-words) aynı pencereyi tek bir örneğe çevirir. Bağlam kelimelerinin girdi vektörlerinin **ortalaması** alınır, merkez kelime bu ortalamadan tahmin edilir:
$$\mathbf h=\frac{1}{|\mathcal C_t|}\sum_{j\in\mathcal C_t}\mathbf v_{w_{t+j}},\qquad p(w_t\mid\mathcal C_t)=\mathrm{softmax}(\mathbf W_{\text{out}}\mathbf h)_{w_t}.$$
Adındaki "bag" (torba), ortalamanın bağlamdaki **sırayı yok saymasından** gelir. "continuous" ise sürekli (yoğun) temsil kullanıldığını vurgular.

:::türetim CBOW gradyanı bağlama eşit dağılır
$J=-\log p(w_t\mid\mathcal C_t)$ ve $\mathbf g=\partial J/\partial\mathbf h$ olsun; skip-gram'daki $\partial J/\partial\mathbf v_c$ ile aynı biçimdedir. Ortalamanın her terime türevi $1/|\mathcal C_t|$ olduğundan
$$\frac{\partial J}{\partial\mathbf v_{w_{t+j}}}=\frac{1}{|\mathcal C_t|}\,\mathbf g\qquad(j\in\mathcal C_t).$$
Her bağlam kelimesi gradyanın eşit bir payını alır. Çıktı tarafı, skip-gram'daki bir çiftle aynı şekilde güncellenir.
:::

:::uyarı Kaynak notu: word2vec.c'nin CBOW güncellemesi
Orijinal C kodu (ve Gensim'in varsayılanı) $\mathbf h$'yi ortalama olarak hesaplar. Ama geri geçişte $1/|\mathcal C_t|$ çarpanını uygulamaz: tam $\mathbf g$'yi **her** bağlam vektörüne ekler. Kod satırı: `syn0[...] += neu1e[...]`. Girdi tarafının adımı gerçek gradyandan $|\mathcal C_t|$ kat büyük olur, çıktı tarafınınki olmaz; güncelleme yönü stokastik gradyandan sapar. İrsoy, Benton ve Stratos (2020, "kōan") bu düzeltme yapıldığında CBOW'un skip-gram kadar iyi gömmeler öğrendiğini ve birkaç kat hızlı kaldığını raporladı. "CBOW anlamsal analojilerde zayıftır" bulgusunun bir kısmı bu uygulama ayrıntısından kaynaklanıyor olabilir.
:::

**Hangisi?** Mikolov ve ark. (2013a, Tablo 3) aynı veri ve 640 boyutla dört mimariyi karşılaştırdı. Analoji testinde anlamsal / sözdizimsel doğruluk (%) şöyleydi: RNN dil modeli 9 / 36, ileri beslemeli sinirsel dil modeli (NNLM) 23 / 53, **CBOW 24 / 64**, **skip-gram 55 / 59**. Skip-gram anlamsal ilişkilerde belirgin üstündü, CBOW sözdizimselde biraz iyiydi. Makaledeki eğitim karmaşıklığı (eğitim örneği başına) hiyerarşik softmax ile şöyledir:
$$Q_{\text{CBOW}}=N\times D+D\times\log_2V,\qquad Q_{\text{SG}}=C\times(D+D\times\log_2V).$$
Burada $N$ bağlam kelime sayısı, $D$ embedding boyu, $C$ en büyük penceredir. Skip-gram her bağlam kelimesi için ayrı bir çıktı hesabı yaptığı için ≈ $C$ kat pahalıdır. Pratik kural (Mikolov'un word2vec notları): skip-gram nadir kelimelerde daha iyi, CBOW daha hızlıdır. `word2vec.c`'nin varsayılanı CBOW'dur (`-cbow 1`, başlangıç öğrenme oranı 0.05; skip-gram'da 0.025).

### Hiyerarşik softmax {#hs}

Tam softmax'ın $O(V)$ paydasından kurtulmanın ilk yolu, Mikolov ve ark.'nın 2013a'da kullandığı **hiyerarşik softmax**'tır (Morin ve Bengio 2005). Kelimeler bir ikili ağacın $V$ yaprağına yerleştirilir. Her **iç düğüm** $n$'nin (toplam $V-1$ tane) kendi vektörü $\mathbf u_n$ vardır. Bir kelimenin olasılığı, kökten o kelimenin yaprağına giden yoldaki sol/sağ kararlarının çarpımıdır:
$$p(w\mid c)=\prod_{j=1}^{L(w)}\sigma\big(s_j\,\mathbf u_{n_j}^\top\mathbf v_c\big),\qquad s_j=\begin{cases}+1&\text{yol sola gidiyorsa (bit 0)}\\-1&\text{sağa (bit 1)}\end{cases}$$

:::kanıt Olasılıkların toplamı 1'dir
Her iç düğümde iki çocuğa giden olasılıklar $\sigma(x)$ ve $\sigma(-x)$'tir ve $\sigma(x)+\sigma(-x)=1$'dir. Tümevarım: bir alt ağacın yapraklarındaki olasılıkların toplamı, o alt ağacın köküne ulaşma olasılığına eşittir. Yaprak için bu tanım gereğidir. İç düğüm $n$ için sol alt ağacın toplamı $P(n)\sigma(x_n)$, sağınki $P(n)\sigma(-x_n)$'dir ve toplamları $P(n)$'dir. Kökte $P=1$ olduğundan tüm yaprakların toplamı 1'dir. Ayrı bir normalizasyon gerekmez.
:::

Maliyet, yol uzunluğu $L(w)$ kadar iç çarpımdır. Dengeli bir ağaçta $\lceil\log_2V\rceil$ olur. Mikolov ve ark. **Huffman ağacı** kullandı: sık kelimelere kısa kod verilir. Bu durumda frekans ağırlıklı ortalama yol uzunluğu, unigram dağılımının **entropisinin** en fazla 1 bit üstündedir (kaynak kodlama teoremi). Zipf dağılımlı sözlüklerde bu, $\log_2V$'den belirgin kısadır (Şekil 2).

<figure class="fig wide"><svg viewBox="0 0 640 272" role="img" aria-label="Tam softmax, hiyerarşik softmax ve negatif örneklemenin çift başına iç çarpım sayısı sözlük boyuyla nasıl değişir: tam softmax doğrusal, hiyerarşik softmax logaritmik, negatif örnekleme sabit"><line x1="90" y1="230" x2="430" y2="230" class="ink"/><line x1="90" y1="30" x2="90" y2="230" class="ink"/><line x1="90" y1="230.0" x2="430" y2="230.0" class="mute"/><text x="84" y="234.0" text-anchor="end" class="sm">10⁰</text><line x1="90" y1="196.7" x2="430" y2="196.7" class="mute"/><text x="84" y="200.7" text-anchor="end" class="sm">10¹</text><line x1="90" y1="163.3" x2="430" y2="163.3" class="mute"/><text x="84" y="167.3" text-anchor="end" class="sm">10²</text><line x1="90" y1="130.0" x2="430" y2="130.0" class="mute"/><text x="84" y="134.0" text-anchor="end" class="sm">10³</text><line x1="90" y1="96.7" x2="430" y2="96.7" class="mute"/><text x="84" y="100.7" text-anchor="end" class="sm">10⁴</text><line x1="90" y1="63.3" x2="430" y2="63.3" class="mute"/><text x="84" y="67.3" text-anchor="end" class="sm">10⁵</text><line x1="90" y1="30.0" x2="430" y2="30.0" class="mute"/><text x="84" y="34.0" text-anchor="end" class="sm">10⁶</text><text x="90.0" y="246" text-anchor="middle" class="sm">1K</text><text x="203.3" y="246" text-anchor="middle" class="sm">10K</text><text x="316.7" y="246" text-anchor="middle" class="sm">100K</text><text x="430.0" y="246" text-anchor="middle" class="sm">1M</text><text x="260.0" y="264" text-anchor="middle" class="sm">sözlük boyu V (log ekseni)</text><text x="18" y="130.0" transform="rotate(-90 18 130.0)" text-anchor="middle" class="sm">çift başına d-boyutlu iç çarpım</text><path d="M90.0,130.0 L203.3,96.7 L316.7,63.3 L430.0,30.0" class="cb w2" /><circle cx="90.0" cy="130.0" r="3" class="cb-f"/><circle cx="203.3" cy="96.7" r="3" class="cb-f"/><circle cx="316.7" cy="63.3" r="3" class="cb-f"/><circle cx="430.0" cy="30.0" r="3" class="cb-f"/><path d="M90.0,196.7 L203.3,191.8 L316.7,189.0 L430.0,186.6" class="cq dash w2" /><circle cx="90.0" cy="196.7" r="3" class="cq-f"/><circle cx="203.3" cy="191.8" r="3" class="cq-f"/><circle cx="316.7" cy="189.0" r="3" class="cq-f"/><circle cx="430.0" cy="186.6" r="3" class="cq-f"/><path d="M90.0,200.8 L203.3,197.3 L316.7,194.6 L430.0,192.4" class="cq w2" /><circle cx="90.0" cy="200.8" r="3" class="cq-f"/><circle cx="203.3" cy="197.3" r="3" class="cq-f"/><circle cx="316.7" cy="194.6" r="3" class="cq-f"/><circle cx="430.0" cy="192.4" r="3" class="cq-f"/><path d="M90.0,189.9 L203.3,189.9 L316.7,189.9 L430.0,189.9" class="cp dash w2" /><circle cx="90.0" cy="189.9" r="3" class="cp-f"/><circle cx="203.3" cy="189.9" r="3" class="cp-f"/><circle cx="316.7" cy="189.9" r="3" class="cp-f"/><circle cx="430.0" cy="189.9" r="3" class="cp-f"/><path d="M90.0,204.1 L203.3,204.1 L316.7,204.1 L430.0,204.1" class="cp w2" /><circle cx="90.0" cy="204.1" r="3" class="cp-f"/><circle cx="203.3" cy="204.1" r="3" class="cp-f"/><circle cx="316.7" cy="204.1" r="3" class="cp-f"/><circle cx="430.0" cy="204.1" r="3" class="cp-f"/><line x1="452" y1="44" x2="474" y2="44" class="cb w2"/><text x="480" y="48" class="sm">tam softmax: V</text><line x1="452" y1="66" x2="474" y2="66" class="cq dash w2"/><text x="480" y="70" class="sm">HS dengeli: ⌈log₂V⌉</text><line x1="452" y1="88" x2="474" y2="88" class="cq w2"/><text x="480" y="92" class="sm">HS Huffman, Zipf s = 1</text><line x1="452" y1="110" x2="474" y2="110" class="cp dash w2"/><text x="480" y="114" class="sm">NEG-15: k+1 = 16</text><line x1="452" y1="132" x2="474" y2="132" class="cp w2"/><text x="480" y="136" class="sm">NEG-5: k+1 = 6</text><text x="452" y="170" class="sm">V = 10⁶, Huffman:</text><text x="452" y="186" class="sm">ortalama 13.4 (log₂V = 19.9,</text><text x="452" y="202" class="sm">entropi 13.4 bit)</text></svg><figcaption>Şekil 2 — Bir (merkez, bağlam) çifti için gereken d-boyutlu iç çarpım sayısı (ileri geçiş; geri geçiş aynı mertebede). Huffman eğrisi Zipf (s = 1) frekanslarıyla gerçekten kurulan ağaçların frekans ağırlıklı ortalama yol uzunluğudur: V = 1K, 10K, 100K, 1M için 7.5, 9.6, 11.5, 13.4. Kaynak kod çözme teoremi gereği bu değer entropinin en fazla 1 bit üstündedir ve log₂V'den belirgin kısadır. Negatif örnekleme V'den bağımsızdır.</figcaption></figure>

Ek x2'de Huffman ağacını `heapq` ile kurup $p(w\mid c)$'yi yol çarpımı olarak hesaplayacaksın. İkinci makaledeki karşılaştırmada (2013b, Tablo 1; 300 boyutlu skip-gram, ≈ 1 milyar kelime, analoji doğruluğu %) Huffman'lı hiyerarşik softmax 47, negatif örnekleme $k=5$ ile 59, $k=15$ ile 61 verdi. Sık kelime alt-örneklemesiyle bu sayılar sırasıyla 55, 60 ve 61 oldu. Hiyerarşik softmax çok sınıflı büyük çıktılarda hâlâ kullanılır; örneğin fastText'in metin sınıflandırma modu bunu bir kayıp seçeneği olarak sunar (`-loss hs`).

### Negatif örnekleme {#negatif}

İkinci makalenin ana katkısı daha basit bir amaçtır. Mantığı şudur: $V$ sınıflı bir sınıflandırma yapmak yerine her gerçek $(c,o)$ çiftini, rastgele çekilmiş $k$ "gürültü" çiftinden ayıran **ikili lojistik regresyon** yapılır.

Kökeni **gürültü karşıtlıklı kestirim**dir (noise-contrastive estimation, NCE; Gutmann ve Hyvärinen 2010; dil modellerinde Mnih ve Teh 2012). NCE, normalize edilmemiş bir modeli, verinin gürültüden ayırt edilmesi problemine çevirerek eğitir. Uygun koşullarda gerçek olasılıkları kestirir. **Negatif örnekleme** (NEG) NCE'nin sadeleştirilmiş hâlidir: gürültü olasılığını sınıflandırıcıdan çıkarır ve skoru doğrudan $\sigma(\mathbf u^\top\mathbf v)$ yapar. Bunun bedeli, $p(o\mid c)$'nin artık kestirilmemesidir (aşağıdaki uyarı). Kelime vektörü öğrenmek için ise bu kayıp önemsizdir.

:::tanım SGNS amacı (skip-gram + negatif örnekleme)
Her gözlenen $(c,o)$ çifti için gürültü dağılımı $P_n$'den $k$ kelime $w_1,\dots,w_k$ çekilir:
$$J(c,o)=-\log\sigma(\mathbf u_o^\top\mathbf v_c)-\sum_{i=1}^{k}\log\sigma(-\mathbf u_{w_i}^\top\mathbf v_c),\qquad w_i\sim P_n .$$
İlk terim, gerçek çifti "gerçek" (etiket 1) olarak sınıflandırmanın, ikinci terim gürültü çiftlerini "gürültü" (etiket 0) olarak sınıflandırmanın log-kaybıdır; $1-\sigma(x)=\sigma(-x)$. Mikolov ve ark. küçük veri kümelerinde $k=5$–20'yi, büyüklerde 2–5'i önerir.
:::

:::türetim SGNS gradyanları
Kullanacağımız tek özdeşlik: $\dfrac{d}{dx}\log\sigma(x)=1-\sigma(x)$ ve $\dfrac{d}{dx}\log\sigma(-x)=-\sigma(x)$. Zincir kuralıyla:
$$\frac{\partial J}{\partial\mathbf v_c}=\big(\sigma(\mathbf u_o^\top\mathbf v_c)-1\big)\mathbf u_o+\sum_{i=1}^k\sigma(\mathbf u_{w_i}^\top\mathbf v_c)\,\mathbf u_{w_i},$$
$$\frac{\partial J}{\partial\mathbf u_o}=\big(\sigma(\mathbf u_o^\top\mathbf v_c)-1\big)\mathbf v_c,\qquad\frac{\partial J}{\partial\mathbf u_{w_i}}=\sigma(\mathbf u_{w_i}^\top\mathbf v_c)\,\mathbf v_c .$$
Her terim (tahmin − etiket) × (öbür vektör) biçimindedir; M1.4'teki lojistik regresyon gradyanıyla aynı kalıp. Bir negatif, model ona yanlışlıkla "gerçek" dedikçe ($\sigma(\mathbf u_{w_i}^\top\mathbf v_c)\to1$) daha güçlü itilir; zaten ayrılmış bir negatif ($\sigma\to0$) hiçbir şey katmaz. Çift başına maliyet $(k+1)$ iç çarpım ve $(k+2)$ vektör güncellemesidir: $O(kd)$, $V$'den bağımsız. `word2vec.c` bu formülü doğrudan uygular. Tek fark, σ'yı $[-6,6]$ aralığında önceden hesaplanmış bir tablodan okuması ve aralık dışında sabit etiket hatası kullanmasıdır (`MAX_EXP 6`).
:::

:::örnek El hesabı (modülün el hesabı alanlarıyla aynı sayılar)
1 boyutlu vektörler: $\mathbf v_c=1$, $\mathbf u_o=1$, iki negatif $\mathbf u_1=0$, $\mathbf u_2=-1$. Skorlar $\mathbf u_o^\top\mathbf v_c=1$, $\mathbf u_1^\top\mathbf v_c=0$, $\mathbf u_2^\top\mathbf v_c=-1$.
- Kayıp: $J=-\log\sigma(1)-\log\sigma(0)-\log\sigma(1)=0.3133+0.6931+0.3133=\mathbf{1.3197}$. Üçüncü terim $-\log\sigma(-(-1))=-\log\sigma(1)$'dir.
- Gradyan: $\partial J/\partial\mathbf v_c=(0.7311-1)\cdot1+0.5\cdot0+0.2689\cdot(-1)=-0.2689-0.2689=\mathbf{-0.5379}$.
- Çıktı tarafı: $\partial J/\partial\mathbf u_o=(0.7311-1)\cdot1=-0.2689$; $\partial J/\partial\mathbf u_1=0.5$; $\partial J/\partial\mathbf u_2=0.2689$.

Gradyan inişi $\mathbf v_c$'yi büyütür (pozitif yöne: $u_o$'ya doğru, $u_2$'den uzağa). $\mathbf u_1=0$ negatifinin kayıp katkısı ($\log2$) ise $\mathbf v_c$ ne olursa olsun değişmez. Bu terim ancak $\mathbf u_1$'in kendisi güncellenince küçülür.
:::

Aşağıdaki deneyde bu hesabı 2-B vektörlerle yap. Vektörleri değiştir, başlangıç gradyanını oku, gradyan inişinin vektörleri nasıl hareket ettirdiğini izle. "Ders el hesabı" senaryosu yukarıdaki sayıları (1.3197 ve −0.538) verir.

::lab l1

:::yanılgı "Negatif örnekleme softmax'ın yaklaşığıdır"
Değildir. NCE, $k\to\infty$'da ve uygun gürültü düzeltmesiyle softmax'ın gradyanına yakınsar. NEG'in amacı ise farklı bir amaçtır: optimumu $p(o\mid c)$ değil, aşağıda türeteceğimiz **kaydırılmış PMI**'dir (Dyer 2014; Levy ve Goldberg 2014). Bu yüzden SGNS'le eğitilmiş bir modelden dil modeli olasılığı okunamaz. Kelime vektörleri için bu sorun değildir, çünkü hedef olasılık değil geometridir.
:::

### Gürültü dağılımı ve sık kelime alt-örneklemesi {#gurultu}

**Gürültü dağılımı.** Negatifler hangi dağılımdan çekilmeli? Mikolov ve ark. (2013b) düzgün dağılımı, unigram dağılımı $U(w)$'yu ve $U(w)^{3/4}/Z$'yi denedi; $3/4$ kuvveti hem NCE'de hem NEG'de diğer ikisini açıkça geçti. Bu, kuramsal değil **deneysel** bir seçimdir:
$$P_n(w)=\frac{\#(w)^{\alpha}}{\sum_{w'}\#(w')^{\alpha}},\qquad\alpha=0.75 .$$

:::örnek Düzleşme etkisi
İki kelime, sayımlar 81 ve 16. $\alpha=1$'de ilkinin olasılığı $81/97=0.835$. $\alpha=3/4$'te $81^{3/4}=27$, $16^{3/4}=8$ ve olasılık $27/35=\mathbf{0.771}$. Sayımlar 100 ve 1 olsaydı nadir kelimenin payı $1/101=0.0099$'dan $1/(31.6+1)=0.0307$'ye, yani ≈ 3 katına çıkardı. Kuvvet $<1$ olduğu için oranlar sıkışır: sık kelimeler hâlâ en çok örneklenir ama nadir kelimeler de negatif olma şansı bulur.
:::

`word2vec.c` bu dağılımdan $O(1)$'de örneklemek için $10^8$ hücreli bir tablo kurar. Her kelime $P_n(w)$ oranında hücre kaplar; örnekleme rastgele bir hücre okumaktır. PyTorch'ta aynı iş `torch.multinomial(Pn, B*k, replacement=True)` ile yapılır (aşama d).

**Sık kelime alt-örneklemesi.** "ve", "bir", "bu" gibi kelimeler neredeyse her pencereye girer. Milyonlarca kez görüldükleri için vektörleri erkenden oturur ve sonraki güncellemeler az bilgi taşır. Ayrıca merkez kelimenin "ve" ile birlikte geçmesi, "Fransa" ile "Paris"in birlikte geçmesinden çok daha az bilgilidir. Makaledeki çözüm: eğitim metnindeki her token, frekansı $f(w)$ (sayım / toplam token) olmak üzere
$$P_{\text{at}}(w)=1-\sqrt{\frac{t}{f(w)}}\qquad(f(w)>t\text{ ise; değilse }0)$$
olasılıkla **atılır**. $t$ tipik olarak $10^{-5}$'tir. El hesabı: $f=10^{-3}$ için $1-\sqrt{10^{-2}}=\mathbf{0.9}$.

:::türetim Alt-örnekleme frekansı karekökle sıkıştırır
Tutulma olasılığı $\sqrt{t/f}$ ($f>t$). Bir kelimenin beklenen etkin frekansı
$$f'(w)=f(w)\sqrt{\frac{t}{f(w)}}=\sqrt{t\,f(w)} .$$
Sık kelimelerin frekansı karekök ölçeğine iner. Zipf yasasında ($f_r\propto r^{-s}$) log–log grafikte eğim $-s$'den $-s/2$'ye düşer. $f\le t$ olan kelimelere dokunulmaz. Ayrıca token atıldıktan **sonra** pencere kurulduğu için kalan kelimeler arası etkin uzaklık kısalır: alt-örnekleme pencereyi örtük olarak genişletir (Levy, Goldberg ve Dagan 2015).
:::

:::uyarı Kaynak notu: makaledeki formül ile word2vec.c farklı
Yayımlanan kod makaledeki formülü değil, şu **tutma** olasılığını kullanır ($f$: sayım / toplam token):
$$P_{\text{tut}}=\sqrt{\frac{t}{f}}+\frac{t}{f}\quad(\text{1'den büyükse 1}),\qquad\text{varsayılan }t=10^{-3}\ (\texttt{-sample}).$$
$f=10^{-3}$, $t=10^{-5}$ örneğinde $\sqrt{0.01}+0.01=0.11$ tutulur, $0.89$ atılır: makaledeki 0.9'a yakın. **Atılmanın başladığı eşik** ise farklıdır. $x=\sqrt{t/f}$ koyarsak $x+x^2=1$ olur; çözüm $x=(\sqrt5-1)/2\approx0.618$, yani $f=t/x^2=\varphi^2t\approx2.618\,t$ ($\varphi$: altın oran). Kod, $t$ ile $2.6t$ arasındaki kelimeleri tamamen tutar; makaledeki formül bunları da hafifçe atar. Gensim de kodun formülünü ve $10^{-3}$ varsayılanını izler. Makaleyle karşılaştırma yaparken hangi formülün ve hangi $t$'nin kullanıldığı belirtilmelidir.
:::

Alt-örneklemenin etkisi büyüktür. 2013b, Tablo 1'de NEG-5 eğitimi 38 dakikadan 14 dakikaya indi ve doğruluk 59'dan 60'a çıktı. Hiyerarşik softmax'ta doğruluk 47'den 55'e yükseldi. Son ayrıntı: öğrenme oranı başlangıç değerinden (skip-gram 0.025) eğitim boyunca **doğrusal** olarak azaltılır. Alt sınırı başlangıcın $10^{-4}$ katıdır.

Aşağıdaki deneyde Zipf dağılımlı bir sözlükte $\alpha$'nın negatiflerin nereden geldiğini, $t$'nin ve formül seçiminin hangi tokenları attığını ölç.

::lab l2

### SGNS örtük olarak kaydırılmış PMI'yi çarpanlarına ayırır {#pmi}

word2vec "tahmin" eden bir yöntem gibi görünür; LSA gibi yöntemler ise "sayar". Levy ve Goldberg (2014) ikisinin şaşırtıcı derecede yakın olduğunu gösterdi.

**Noktasal karşılıklı bilgi** (pointwise mutual information, PMI), bir kelime–bağlam çiftinin, bağımsız olsalardı beklenenden ne kadar sık birlikte geçtiğini ölçer:
$$\mathrm{PMI}(w,c)=\log\frac{P(w,c)}{P(w)P(c)}=\log\frac{\#(w,c)\,|D|}{\#(w)\,\#(c)} .$$
Burada $\#(w,c)$ çiftin sayısı, $\#(w)=\sum_c\#(w,c)$, $\#(c)=\sum_w\#(w,c)$, $|D|$ toplam çift sayısıdır. PMI = 0 bağımsızlık, pozitif değer çekim, negatif değer kaçınma anlamına gelir.

:::türetim SGNS'in optimumu: $\mathbf w\cdot\mathbf c=\mathrm{PMI}(w,c)-\log k$
Tüm derlem üzerinden SGNS kaybının beklentisini yazalım. $(w,c)$ çifti $\#(w,c)$ kez pozitif olarak görülür. $w$ merkezken çekilen $k\,\#(w)$ negatifin beklenen $k\,\#(w)P_n(c)$ tanesi $c$'dir. $x=\mathbf w\cdot\mathbf c$ (merkez ve bağlam vektörlerinin iç çarpımı) ve $\ell(x)=-\log\sigma(x)$ ile, toplam kayıpta bu çifte ait terim:
$$L_{wc}(x)=\#(w,c)\,\ell(x)+k\,\#(w)\,P_n(c)\,\ell(-x).$$
**Varsayım:** boyut yeterince büyük, yani her $x_{wc}$ diğerlerinden bağımsız seçilebiliyor. O zaman her terimi ayrı ayrı en küçükleriz. $\ell'(x)=-\sigma(-x)$ ve $\frac{d}{dx}\ell(-x)=\sigma(x)$ olduğundan
$$\frac{dL_{wc}}{dx}=-\#(w,c)\,\sigma(-x)+k\,\#(w)P_n(c)\,\sigma(x)=0\ \Longrightarrow\ \frac{\sigma(x)}{\sigma(-x)}=e^{x}=\frac{\#(w,c)}{k\,\#(w)\,P_n(c)} .$$
$L_{wc}$ dışbükey olduğu için bu tek minimumdur. $P_n(c)=\#(c)/|D|$ ($\alpha=1$) için
$$x^*=\log\frac{\#(w,c)\,|D|}{\#(w)\,\#(c)}-\log k=\mathrm{PMI}(w,c)-\log k .$$
**Sonuç:** SGNS, girdisi kaydırılmış PMI olan $V\times V$ matrisi $\mathbf W_{\text{in}}\mathbf W_{\text{out}}^\top$ biçiminde çarpanlarına ayırır.

**Koşullar ve sapmalar.**
1. $d<V$ ise hücreler bağımsız seçilemez. SGNS bu durumda, sık çiftlere daha çok ağırlık veren ağırlıklı bir düşük-rank yaklaşım yapar; SVD'nin eşit ağırlıklı yaklaşımından farklıdır.
2. $\alpha=0.75$ ile $P_n(c)\propto\#(c)^{0.75}$ olur ve hedef, düzleştirilmiş bağlam olasılığıyla hesaplanan PMI'ye kayar. Nadir bağlamların PMI'si aşağı çekilir (Levy, Goldberg ve Dagan 2015'te "bağlam dağılımı düzleştirme").
3. Hiç görülmemiş çiftlerde $\#(w,c)=0$ ve $x^*=-\infty$'dir. Bu yüzden sayım tabanlı karşılığı, negatifleri sıfıra kırpılmış **SPPMI**'dir: $\max(\mathrm{PMI}-\log k,0)$.
:::

:::örnek El hesabı
$\#(w,c)=10$, $\#(w)=100$, $\#(c)=50$, $|D|=10\,000$, $k=5$. $\mathrm{PMI}=\log\frac{10\cdot10\,000}{100\cdot50}=\log20=2.996$. Optimumda $x^*=\log20-\log5=\log4=1.386$. Kontrol: $\sigma(1.386)=0.8$. Çiftin pozitif görülme sayısı 10, negatif görülme beklentisi $5\cdot100\cdot50/10\,000=2.5$; optimumda sınıflandırıcının "gerçek" olasılığı tam $10/(10+2.5)=0.8$'dir. Model, bu çiftin pozitif örneklerde negatiflere göre ne sıklıkla geçtiğini öğrenir.
:::

Bu sonuç sayım tabanlı bir alternatif verir (ek x1): eş-geçme matrisini say, PPMI ya da SPPMI'yi hesapla, kesik SVD al ve $\mathbf W=\mathbf U_d\boldsymbol\Sigma_d^{1/2}$'yi gömme olarak kullan. "Saymak mı tahmin etmek mi?" tartışmasının iki ucu şöyle: Baroni ve ark. (2014) tahmin tabanlı gömmeleri açıkça üstün buldu. Levy, Goldberg ve Dagan (2015) ise word2vec'in "küçük" ayarlarını (dinamik pencere, alt-örnekleme, $\alpha=0.75$ düzleştirme, $\mathbf W+\mathbf C$) sayım yöntemlerine de taşıyınca farkın çoğunun kapandığını gösterdi. Algoritmadan çok bu hiperparametreler önemliydi.

Aşağıdaki deneyde sentetik derlemin çift sayımlarından SGNS'in **beklenen** amacını doğrudan en küçükle ve öğrenilen $\mathbf w\cdot\mathbf c$'yi hedef $x^*$ ile karşılaştır. $d$'yi küçültünce rank kısıtının nasıl devreye girdiğini, SPPMI'nin tekil değer spektrumunda gör.

::lab l3

### GloVe: log eş-geçmeyi doğrudan modellemek {#glove}

Pennington, Socher ve Manning (2014), PMI bağlantısını açık bir amaca dönüştürdü. $X_{ij}$, $j$'nin $i$'nin bağlamında kaç kez geçtiğini gösterir (uzaklığa göre $1/\delta$ ağırlıklı). $P_{ij}=X_{ij}/X_i$ olsun.

:::sezgi Oranlar anlam taşır
"kral" ve "kraliçe"yi ayıran şeyi bulmak için $P(\text{deneme kelimesi}\mid\text{kral})/P(\text{deneme kelimesi}\mid\text{kraliçe})$ oranına bakalım. "bıyık" için oran büyük, "etek" için küçüktür. "taç" ya da "sokak" (ikisiyle de eşit ilgili ya da ilgisiz) için ≈ 1'dir. Ham olasılıklar sık/nadir etkisiyle karışıktır; **oranlar** ise iki kelimenin farkını gösterir. GloVe bu oranı vektör farkıyla ifade etmek ister: $F\big((\mathbf w_i-\mathbf w_j)^\top\tilde{\mathbf w}_k\big)=P_{ik}/P_{jk}$.
:::

:::türetim Oran koşulundan GloVe amacına
Farkın oranla eşleşmesi ve toplamanın çarpmaya dönüşmesi istenirse $F=\exp$ seçilir. Bu durumda $\mathbf w_i^\top\tilde{\mathbf w}_k=\log P_{ik}=\log X_{ik}-\log X_i$ olmalıdır. $\log X_i$ terimi $k$'ya bağlı olmadığından bir bias $b_i$'ye katılır. Simetri için bir de $\tilde b_k$ eklenir:
$$\mathbf w_i^\top\tilde{\mathbf w}_k+b_i+\tilde b_k=\log X_{ik}.$$
Bu eşitliği yalnız $X_{ik}>0$ hücrelerinde ağırlıklı en küçük kareyle zorlamak GloVe amacıdır:
$$J=\sum_{i,k:\,X_{ik}>0}f(X_{ik})\big(\mathbf w_i^\top\tilde{\mathbf w}_k+b_i+\tilde b_k-\log X_{ik}\big)^2,\qquad f(x)=\begin{cases}(x/x_{\max})^{\alpha}&x<x_{\max}\\1&\text{aksi}\end{cases}$$
Makalede $x_{\max}=100$, $\alpha=3/4$ kullanılır; yazarlar word2vec'in $3/4$'üyle aynı değerin deneysel olarak en iyi çıkmasını ilginç bulduklarını not eder. $f$, nadir (gürültülü) sayımlara az, çok sık sayımlara sınırlı ağırlık verir. Örnek: $f(10)=0.1^{0.75}=0.178$. **PMI bağlantısı:** biaslar $\log X_i$ ve $\log X_k$'yı (artı sabit) emerse $\mathbf w_i^\top\tilde{\mathbf w}_k\approx\log X_{ik}-\log X_i-\log X_k+\text{sabit}$ olur, yani PMI. GloVe, SGNS'in örtük yaptığını açıkça yapar. Fark, kaybın biçimindedir: kare kayıp ve log sayım.
:::

GloVe eğitimden sonra $\mathbf W+\tilde{\mathbf W}$'yi gömme olarak kullanır; iki tablo simetrik rol oynadığı için toplam gürültüyü azaltır. Eğitim, sıfır olmayan hücreler üzerinde AdaGrad ile yapılır. Maliyet, derlem boyuna değil sıfır olmayan $X$ hücre sayısına bağlıdır.

### fastText: alt-kelime n-gramlarıyla vektörler {#fasttext}

Slaytlardaki alt-kelime tartışması (karakter düzeyi embedding'ler, BPE) M4.5'te ayrıntılı işlendi. Kelime vektörleri tarafındaki karşılığı **fastText**'tir (Bojanowski, Grave, Joulin, Mikolov; TACL 2017). Bir kelime, sınır işaretleriyle `<` ve `>` arasına alınır. Kelime, $3\le n\le6$ uzunluklu tüm karakter n-gramlarının ve kelimenin kendisinin kümesi $\mathcal G_w$ ile temsil edilir:
$$s(w,c)=\sum_{g\in\mathcal G_w}\mathbf z_g^\top\mathbf u_c .$$
Kelime vektörü n-gram vektörlerinin **toplamıdır**. Ek x3 ortalamayı kullanıyor; bu yalnız bir ölçek farkı yaratır, kosinüs benzerliğini etkilemez. N-gramlar FNV-1a karmasıyla $K=2\times10^6$ kovaya eşlenir; bu, OOV kovası fikrinin n-gram düzeyine taşınmasıdır.

:::örnek "kedi"nin n-gramları
`<kedi>` dizgisi 6 karakterdir. $n=3$: `<ke`, `ked`, `edi`, `di>` (4); $n=4$: `<ked`, `kedi`, `edi>` (3); $n=5$: `<kedi`, `kedi>` (2); $n=6$: `<kedi>` (1). Toplam 10 öğe vardır ve 6-gram zaten kelimenin kendisidir. "kedilerimiz" hiç görülmemiş olsa bile `<ke`, `ked`, `edi`, `<kedi` gibi n-gramları paylaştığı için anlamlı bir vektör alır. Makale Türkçeyi açıkça zengin biçimbilimli diller arasında sayar; ekleme dayalı dillerde kazanç en büyüktür.
:::

Karşılaştırma: BPE (M4.5) tokenları **ayırır** ve her parçaya ayrı vektör verir; dizi modeli bunları birleştirir. fastText ise parçaları **toplayarak** kelime düzeyinde tek bir vektör üretir. İkisi de OOV sorununu çözer. Karakter düzeyi embedding'lerin slayttaki sorunu (5 kelimelik cümlenin 30–40 vektöre dönüşmesi) fastText'te dizi uzunluğuna yansımaz, çünkü n-gram vektörleri kelime başına toplanır. Bedeli kelime başına ≈ 10–40 vektör okumaktır.

### Analojiler ve 3CosAdd {#analoji}

Slayttaki iki ünlü örnek, word2vec uzayının yalnız yakınlığı değil **ilişkileri** de kodladığını gösterir:
$$\text{kral}-\text{adam}+\text{kadın}\approx\text{kraliçe},\qquad\text{Madrid}-\text{İspanya}+\text{Fransa}\approx\text{Paris}.$$
Slayt 13'teki şekil, ikinci makaledeki (2013b, Şekil 2) **1000 boyutlu** skip-gram vektörlerinin ülkeleri ve başkentleri için PCA ile 2-B'ye indirilmiş hâlidir. Ülke→başkent okları kabaca paraleldir. Slaytın dediği gibi, "başkent" ilişkisine karşılık gelen fark vektörü bu indirgenmiş uzayda soldan sağa uzanan yatay bir yöne düşer.

:::uyarı Kaynak düzeltmesi: "capitol"
Slayt 13'te "the semantic vector component that corresponds to *capitol*" yazıyor. Burada kastedilen **capital**'dir (başkent). *Capitol* İngilizcede yasama meclisinin binası anlamına gelir (ör. ABD Kongre binası). Doğru ifade: "başkent ilişkisine karşılık gelen vektör bileşeni".
:::

:::uyarı Kaynak notu: 2-B PCA çizimlerindeki paralellik
PCA çizimi seçilmiş az sayıda kelime üzerinde yapılır ve 1000 boyuttan 2 boyuta inerken mesafeler bozulur. Paralel görünen okların kosinüs benzerliği yüksek boyutta ölçülmelidir. Aynı izdüşüme başka kelimeler eklendiğinde düzen kaybolabilir. Şekil, ilişkinin varlığının bir **gösterimidir**, kanıtı değildir. Kanıt aşağıdaki nicel analoji testidir.
:::

**Test seti ve ölçüt.** Mikolov ve ark. (2013a) 8 869 anlamsal (başkent–ülke, para birimi, aile ilişkisi…) ve 10 675 sözdizimsel (çoğul, zaman, karşılaştırma…) olmak üzere toplam 19 544 soruluk bir set yayımladı ("Google analogy"). Her soru "$a$'nın $b$'ye oranı, $c$'nin ?'ye oranı gibidir" biçimindedir. Cevap yalnız **tam isabetle** (ilk sıradaki kelime doğruysa) doğru sayılır; eşanlamlılar yanlış kabul edilir.

:::türetim 3CosAdd: üç kosinüsün toplamı
Cevap $x^*=\arg\max_{x\notin\{a,b,c\}}\cos(\mathbf x,\ \mathbf b-\mathbf a+\mathbf c)$ (yukarıdaki örnekte $a$ = adam, $b$ = kral, $c$ = kadın). Vektörler birim uzunluğa normalize edilmiş olsun ($\hat{\mathbf x}$). Sorgu vektörünün normu $\|\hat{\mathbf b}-\hat{\mathbf a}+\hat{\mathbf c}\|$, $x$'e bağlı değildir. Bu yüzden
$$\arg\max_x\cos(\hat{\mathbf x},\hat{\mathbf b}-\hat{\mathbf a}+\hat{\mathbf c})=\arg\max_x\big(\cos(\mathbf x,\mathbf b)-\cos(\mathbf x,\mathbf a)+\cos(\mathbf x,\mathbf c)\big).$$
Yani "vektör aritmetiği", aslında $b$'ye ve $c$'ye benzeyen, $a$'ya benzemeyen kelimeyi arayan **toplamsal bir puanlamadır** (Levy ve Goldberg 2014b). Zayıf noktası, bir terimin baskın çıkabilmesidir: $\mathbf x$, $\mathbf b$'ye çok benziyorsa diğer iki terim fark etmez. Önerilen **3CosMul** çarpımsal birleştirir: $\dfrac{\cos'(\mathbf x,\mathbf b)\cos'(\mathbf x,\mathbf c)}{\cos'(\mathbf x,\mathbf a)+\varepsilon}$; burada $\cos'=(\cos+1)/2\in[0,1]$. Bu biçim aynı vektörlerle analoji doğruluğunu artırır.
:::

:::yanılgı "Vektör aritmetiği doğrudan cevabı verir"
Standart değerlendirme girdi kelimelerini ($a$, $b$, $c$) aday listesinden **çıkarır**. Çıkarmazsa, gerçek gömmelerde $\mathbf b-\mathbf a+\mathbf c$'ye en yakın kelime çoğu zaman $\mathbf b$'nin (ya da $\mathbf c$'nin) kendisidir. Fark vektörü $\mathbf c-\mathbf a$ genellikle kısadır ve $\mathbf b$'yi yerinden pek oynatmaz (Linzen 2016; Nissim, van Noord ve van der Goot 2020). Yani "kral − adam + kadın = kraliçe" sonucu, "kral" seçeneği elendikten sonraki en yakın komşudur. Bu kısıt, yanlılık analizlerinde de yanıltıcı sonuçlara yol açabilir: cevabın girdiyle aynı olması yasaklandığı için "doktor − erkek + kadın" sorgusu "doktor"u döndüremez.
:::

Aşağıdaki şekil ve deney küçük bir sentetik derlemde çalışır: her varlık kelimesi yalnız kendi özelliklerine (cinsiyet, statü, yaş; şehir/ülke, dil) özgü bağlam kelimeleriyle geçer. Bu yüzden ilişkiler temizdir ve girdi kelimeleri hariç tutulmasa da kraliçe birinci gelir. Şekil 3'teki listeler bunu gösteriyor. Gerçek metin çok daha gürültülüdür.

<figure class="fig wide"><svg viewBox="0 0 640 310" role="img" aria-label="Eğitilmiş on altı boyutlu kelime vektörlerinin PCA izdüşümü; erkek-kadın çiftlerini birleştiren oklar kabaca paralel, kral eksi adam artı kadın noktası kraliçeye düşüyor"><defs><marker id="f3a" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker><marker id="f3b" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><line x1="153.4" y1="233.2" x2="246.7" y2="280.1" class="cb w2" marker-end="url(#f3b)"/><line x1="246.8" y1="117.1" x2="344.5" y2="158.3" class="cb w2" marker-end="url(#f3b)"/><line x1="54.7" y1="165.5" x2="141.7" y2="209.0" class="cb" marker-end="url(#f3b)"/><line x1="150.5" y1="39.9" x2="236.0" y2="88.3" class="cb" marker-end="url(#f3b)"/><line x1="153.4" y1="233.2" x2="246.8" y2="117.1" class="ca dash" marker-end="url(#f3a)"/><line x1="246.7" y1="280.1" x2="340.2" y2="163.9" class="ca dash" marker-end="url(#f3a)"/><circle cx="246.8" cy="117.1" r="4.5" class="ca-f"/><text x="240.8" y="109.1" text-anchor="end">kral</text><circle cx="344.5" cy="158.3" r="4.5" class="ca-f"/><text x="352.5" y="148.3" text-anchor="start">kraliçe</text><circle cx="153.4" cy="233.2" r="4.5" class="cp-f"/><text x="145.4" y="237.2" text-anchor="end">adam</text><circle cx="246.7" cy="280.1" r="4.5" class="cp-f"/><text x="254.7" y="294.1" text-anchor="start">kadın</text><circle cx="150.5" cy="39.9" r="4.5" class="ca-f"/><text x="156.5" y="31.9" text-anchor="start">prens</text><circle cx="236.0" cy="88.3" r="4.5" class="ca-f"/><text x="244.0" y="92.3" text-anchor="start">prenses</text><circle cx="54.7" cy="165.5" r="4.5" class="cp-f"/><text x="54.7" y="155.5" text-anchor="middle">oğlan</text><circle cx="141.7" cy="209.0" r="4.5" class="cp-f"/><text x="133.7" y="203.0" text-anchor="end">kız</text><circle cx="340.2" cy="163.9" r="7" class="cq w2"/><text x="350.2" y="181.9" class="tq">q</text><text x="420" y="40" class="tq">q = kral − adam + kadın</text><text x="420" y="64" class="sm">en yakın (a, b, c hariç), kosinüs:</text><text x="420" y="84" class="sm">1. kraliçe — 0.987</text><text x="420" y="101" class="sm">2. prenses — 0.674</text><text x="420" y="118" class="sm">3. taht — 0.515</text><text x="420" y="150" class="sm">hariç tutmadan:</text><text x="420" y="170" class="sm">1. kraliçe — 0.987</text><text x="420" y="187" class="sm">2. kadın — 0.703</text><text x="420" y="204" class="sm">3. kral — 0.681</text><text x="420" y="221" class="sm">4. prenses — 0.674</text><text x="420" y="252" class="sm">turuncu oklar: cinsiyet farkı</text><text x="420" y="269" class="sm">mavi kesikli: statü farkı</text><text x="420" y="286" class="sm">(d = 16 → PCA ile 2-B)</text></svg><figcaption>Şekil 3 — 3CosAdd analojisi, M4.12 deneyindeki derlemle eğitilmiş 16 boyutlu SGNS vektörleri üzerinde (tohum 1, varsayılan ayarlar), PCA ile 2-B'ye indirilmiş. Turuncu oklar (adam→kadın, kral→kraliçe, oğlan→kız, prens→prenses) kabaca paraleldir. q noktası, adam→kral farkının kadın'a eklenmesidir ve kraliçe'nin hemen yanına düşer. Bu temiz derlemde girdi kelimeleri hariç tutulmasa da kraliçe kazanıyor; gerçek gömmelerde çoğu zaman kazanmaz (bkz. metin). PCA uzaklıkları yalnız yaklaşıktır: kosinüsler 16 boyutta hesaplandı.</figcaption></figure>

Deneyde boyut $d$, pencere, negatif sayısı $k$, epoch, gürültü üssü $\alpha$ ve alt-örneklemeyi değiştirip analojilerin ne zaman ortaya çıktığını, ne zaman bozulduğunu gözle. Vektörler 2-B'ye PCA ile indirilir; analoji ve kosinüsler tam boyutta hesaplanır.

::lab l4

### Yanlılık ve statik vektörlerin sınırları {#sinirlar}

**Toplumsal yanlılık.** Gömmeler eğitim metnindeki istatistikleri öğrenir; kalıp yargılar da bu istatistiklerin parçasıdır. Bolukbasi ve ark. (2016), Google News word2vec vektörlerinde meslek–cinsiyet ilişkilerinin analojilerle görünür olduğunu gösterdi; ünlü bir örnekte programcılığın kadın karşılığı olarak ev işleriyle ilgili bir kelime çıkıyordu. Ayrıca bir "cinsiyet yönü" tanımlayıp onu cinsiyetten bağımsız olması gereken kelimelerden çıkarmayı önerdiler. Caliskan, Bryson ve Narayanan (2017, Science), psikolojideki örtük çağrışım testinin gömme karşılığını (WEAT) kurdu. İnsan deneylerindeki bilinen yanlılıkların GloVe vektörlerinde de ölçülebildiğini gösterdiler. Gonen ve Goldberg (2019), yön çıkarma yöntemlerinin yanlılığı **gizlediğini** ama gidermediğini buldu: "arındırılmış" vektörlerin kümelenmesi hâlâ cinsiyeti ele veriyordu. Türkçenin cinsiyetsiz "o" zamiri burada ilginç bir test durumudur: bir çeviri sistemi "o bir doktor" cümlesini İngilizceye çevirirken cinsiyeti eğitim istatistiklerinden seçmek zorunda kalır.

**Çok anlamlılık.** Statik gömme her kelimeye **tek** bir vektör verir. "yüz" (sayı, surat, yüzmek fiili) ya da "çay" (içecek, akarsu) için bu vektör, anlamların derlemdeki sıklıklarıyla ağırlıklı bir karışımıdır. Slaytın son sayfası buna işaret eder: bağlamsal embedding'ler, başlangıç embedding'lerini dikkat mekanizmasıyla **cümleye göre dönüştürür** (ELMo, BERT, GPT; M4.16). Bugün kullanılan dil modellerinin giriş katmanı yine bir `nn.Embedding` tablosudur, ama asıl temsil bağlamdan sonra oluşur.

**Dağılımsal benzerlik ≠ anlam.** Zıt anlamlılar ("sıcak" ve "soğuk") aynı bağlamlarda geçtiği için vektör uzayında yakın çıkar. Model "ilişkili"yi "benzer"den ayırmaz. Nadir kelimelerin vektörleri az güncelleme aldığı için gürültülüdür. Bu yüzden `min_count` eşiği (`word2vec.c`'de varsayılan 5) altındaki kelimeler sözlükten atılır.

### PyTorch'ta {#pytorch}

| kavram | PyTorch | not |
|---|---|---|
| embedding tablosu | `nn.Embedding(V, d)` | girdi tamsayı tensörü, çıktı `(..., d)` |
| one-hot + biassız katman | `nn.Linear(V, d, bias=False)`, ağırlığı `emb.weight.T` | aşama d: `embedding_as_linear` |
| iki tablo (SGNS) | `inp = nn.Embedding(V, d)`, `out = nn.Embedding(V, d)` | `out` sıfırla başlatılabilir (`word2vec.c` öyle yapar) |
| skor | `(inp(c) * out(o)).sum(-1)`; negatifler için `torch.bmm(out(neg), inp(c)[..., None])` | `(B,)` ve `(B, k)` |
| kayıp | `-(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()` | `logsigmoid` sayısal olarak kararlı |
| gürültü örnekleme | `torch.multinomial(counts**0.75, B*k, replacement=True)` | dağılımı bir kez hesapla |
| seyrek gradyan | `nn.Embedding(..., sparse=True)` + `torch.optim.SparseAdam` | yalnız kullanılan satırlar güncellenir |
| CBOW bağlam ortalaması | `nn.EmbeddingBag(V, d, mode="mean")` ya da `padding_idx` + maske | eksik bağlamlar dolgu indeksiyle |
| tam softmax (karşılaştırma) | `F.cross_entropy(inp(c) @ out.weight.T, o)` | aşama e: $V$ büyüdükçe süre doğrusal artar |
| önceden eğitilmiş vektör | `nn.Embedding.from_pretrained(E, freeze=True)` | ince ayar için `freeze=False` |
| kosinüs komşu | `F.normalize(E, dim=1) @ F.normalize(q, dim=0)`, sonra `topk` | girdi kelimelerini `-inf` ile maskele |

Aşamalar bu sırayı izler. b: saf Python'da sözlük, OOV karması, çift üretimi ve tek çift için kayıp/gradyan (yukarıdaki el hesabı). c: NumPy'da `np.add.at` ile döngüsüz toplu SGNS ve analoji ölçümü. d: PyTorch'ta `nn.Embedding`, seyrek gradyanlar ve CBOW. e: tam softmax ile negatif örneklemenin süre karşılaştırması.

### Özet {#ozet}

- One-hot vektörler diktir: her çift arası $\sqrt2$, iç çarpım 0. OOV kovaları görülmemiş değerleri karmayla toplar; $m$ değer, $b$ kovada beklenen çakışan çift $m(m-1)/2b$.
- One-hot + aktivasyonsuz, biassız yoğun katman = embedding katmanı (20 × 10 → 200 parametre); embedding katmanı aynı hesabı satır okumayla yapar.
- Dağılımsal hipotez: benzer bağlam, benzer anlam. LSA (1990) → NNLM (2003) → word2vec (2013) → GloVe (2014) → fastText.
- Skip-gram: merkezden her bağlam kelimesini softmax ile tahmin et; iki ayrı tablo. Tam softmax gradyanı $\mathbb E_p[\mathbf u]-\mathbf u_o$, çift başına $O(Vd)$. Dinamik pencere uzaklık $\delta$'ya $(C-\delta+1)/C$ ağırlığı verir.
- CBOW: bağlam ortalamasından merkez; gradyan bağlama $1/|\mathcal C|$ ile dağılır. `word2vec.c` bu çarpanı uygulamaz (kōan, 2020).
- Hiyerarşik softmax: ikili ağaçta yol olasılıklarının çarpımı, toplam 1. Huffman ağacında ortalama yol ≈ entropi < $\log_2V$.
- SGNS: $-\log\sigma(\mathbf u_o^\top\mathbf v_c)-\sum_k\log\sigma(-\mathbf u_k^\top\mathbf v_c)$. Gradyanlar (tahmin − etiket) × öbür vektör; maliyet $O(kd)$. El örneği: $J=1.3197$, $\partial J/\partial v_c=-0.538$.
- Gürültü $P_n\propto\#^{3/4}$ (deneysel seçim). Alt-örnekleme makalede $1-\sqrt{t/f}$; kodda tutma $\sqrt{t/f}+t/f$, eşik $\varphi^2t$. Etkin frekans $\sqrt{tf}$.
- SGNS'in optimumu $\mathbf w\cdot\mathbf c=\mathrm{PMI}-\log k$ (yeterli boyut ve $\alpha=1$ koşuluyla); sayım karşılığı SPPMI + SVD. GloVe log sayımları ağırlıklı en küçük kareyle açıkça modeller.
- fastText: kelime = n-gram ($3\le n\le6$) vektörlerinin toplamı; görülmemiş kelimelere de vektör verir.
- Analoji: 3CosAdd = $\cos(x,b)-\cos(x,a)+\cos(x,c)$; girdi kelimeleri hariç tutulur, aksi hâlde çoğu zaman $b$ döner. Slayt düzeltmeleri: Mikolov (Milokov değil), capital (capitol değil).
- Sınırlar: yanlılık, tek vektörde çok anlamlılık (bağlamsal gömmeler M4.16), zıt anlamlıların yakınlığı.
