### Fikir: birçok zayıf model, bir güçlü model {#fikir}

M2.4'te tek bir karar ağacının iki zaafını gördük: derin ağaç **yüksek varyanslıdır** (veriyi biraz değiştir, bambaşka ağaç çıkar), sığ ağaç **yüksek yanlılıklıdır** (eğik sınırı kaba bir merdivenle yaklaştırır). Topluluk (ensemble) yöntemleri birçok modeli birleştirerek bu iki sorunu ayrı ayrı çözer:

| | bagging / rastgele orman | boosting |
|---|---|---|
| hedef | **varyansı** düşür | **yanlılığı** düşür |
| temel model | derin, düşük yanlılıklı ağaçlar | sığ, yüksek yanlılıklı ağaçlar (stump'lar) |
| eğitim | **paralel**, bağımsız | **sıralı**, her model öncekilerin hatasına bakar |
| birleştirme | ortalama / oy | ağırlıklı toplam |
| aşırı uyum | B arttıkça artmaz | tur sayısıyla artabilir (erken durdur) |

<figure class="fig wide"><svg viewBox="0 0 640 210" role="img" aria-label="Bagging: paralel bootstrap ağaçları ortalanır. Boosting: ağaçlar sırayla eklenir, her biri artıklara oturtulur"><g class="sm"><text x="150" y="16" text-anchor="middle">Bagging (paralel)</text><rect x="110" y="26" width="80" height="24" rx="5" class="soft-a"/><text x="150" y="42" text-anchor="middle">veri</text><g class="ink"><line x1="130" y1="50" x2="40" y2="80"/><line x1="140" y1="50" x2="110" y2="80"/><line x1="160" y1="50" x2="190" y2="80"/><line x1="170" y1="50" x2="260" y2="80"/></g><rect x="10" y="80" width="60" height="22" rx="4" class="mute" fill="none"/><text x="40" y="95" text-anchor="middle">boot. 1</text><rect x="80" y="80" width="60" height="22" rx="4" class="mute" fill="none"/><text x="110" y="95" text-anchor="middle">boot. 2</text><text x="150" y="95" text-anchor="middle">…</text><rect x="160" y="80" width="60" height="22" rx="4" class="mute" fill="none"/><text x="190" y="95" text-anchor="middle">boot. 3</text><rect x="230" y="80" width="60" height="22" rx="4" class="mute" fill="none"/><text x="260" y="95" text-anchor="middle">boot. B</text><g class="ink"><line x1="40" y1="102" x2="40" y2="120"/><line x1="110" y1="102" x2="110" y2="120"/><line x1="190" y1="102" x2="190" y2="120"/><line x1="260" y1="102" x2="260" y2="120"/></g><text x="40" y="134" text-anchor="middle">ağaç₁</text><text x="110" y="134" text-anchor="middle">ağaç₂</text><text x="190" y="134" text-anchor="middle">ağaç₃</text><text x="260" y="134" text-anchor="middle">ağaç<tspan baseline-shift="-3" font-size="75%">B</tspan></text><g class="ink"><line x1="40" y1="140" x2="140" y2="170"/><line x1="110" y1="140" x2="145" y2="170"/><line x1="190" y1="140" x2="155" y2="170"/><line x1="260" y1="140" x2="160" y2="170"/></g><rect x="100" y="170" width="100" height="24" rx="5" class="soft-p"/><text x="150" y="186" text-anchor="middle">ortalama</text></g><g class="sm" transform="translate(330,0)"><text x="150" y="16" text-anchor="middle">Boosting (sıralı)</text><rect x="0" y="40" width="56" height="24" rx="5" class="soft-a"/><text x="28" y="56" text-anchor="middle">F₀</text><line x1="56" y1="52" x2="80" y2="52" class="ink"/><rect x="80" y="40" width="56" height="24" rx="5" class="soft-a"/><text x="108" y="56" text-anchor="middle">F₁</text><line x1="136" y1="52" x2="160" y2="52" class="ink"/><rect x="160" y="40" width="56" height="24" rx="5" class="soft-a"/><text x="188" y="56" text-anchor="middle">F₂</text><line x1="216" y1="52" x2="240" y2="52" class="ink"/><text x="262" y="56">… F<tspan baseline-shift="-3" font-size="75%">M</tspan></text><rect x="62" y="100" width="56" height="24" rx="5" class="soft-b"/><text x="90" y="116" text-anchor="middle">h₁</text><rect x="142" y="100" width="56" height="24" rx="5" class="soft-b"/><text x="170" y="116" text-anchor="middle">h₂</text><line x1="90" y1="100" x2="80" y2="64" class="cb"/><line x1="170" y1="100" x2="160" y2="64" class="cb"/><text x="40" y="150">her hₘ, F<tspan baseline-shift="-3" font-size="75%">m−1</tspan>'in artıklarına</text><text x="40" y="166">(negatif gradyana) oturtulur;</text><text x="40" y="182">Fₘ = F<tspan baseline-shift="-3" font-size="75%">m−1</tspan> + η hₘ</text></g></svg><figcaption>Şekil 1 — Bagging'de ağaçlar birbirinden habersiz, farklı bootstrap örneklemlerinde eğitilip ortalanır. Boosting'de her yeni ağaç, şimdiye kadarki toplamın yaptığı hatayı düzeltmeye çalışır.</figcaption></figure>

### Bagging: ortalama almak varyansı düşürür {#bagging}

**Bootstrap** (M0.4): $n$ örnekten yerine koyarak $n$ örnek çek (her örneklemde verinin ~%63'ü bulunur, gerisi tekrardır). $B$ bootstrap örnekleminin her birinde bir model eğit, tahminleri ortala (sınıflandırmada oy). **B**ootstrap **agg**regat**ing** = bagging (Breiman 1996).

:::türetim Ortalamanın varyansı: ρσ² + (1 − ρ)σ²/B
$B$ model, her birinin varyansı $\sigma^2$, herhangi ikisi arasındaki korelasyon $\rho$ (aynı veriden çekildikleri için ilişkililer). $\bar f=\frac1B\sum_bf_b$:
$$\mathrm{Var}[\bar f]=\frac1{B^2}\Big[\sum_b\mathrm{Var}f_b+\sum_{b\ne b'}\mathrm{Cov}(f_b,f_{b'})\Big]=\frac1{B^2}\big[B\sigma^2+B(B-1)\rho\sigma^2\big]=\rho\sigma^2+\frac{1-\rho}B\sigma^2.$$
**Sonuçlar:** (1) $B\to\infty$'da varyans $\rho\sigma^2$'ye iner, sıfıra değil: ortak kısım ortalamayla gitmez. (2) Daha fazla ağaç **asla zarar vermez** (aşırı uyum yapmaz), yalnız hesap maliyeti artar; belli bir $B$'den sonra kazanç önemsizleşir. (3) Yanlılık değişmez: $\mathbb E[\bar f]=\mathbb E[f_b]$. Bagging bu yüzden düşük yanlılıklı, yüksek varyanslı modellerle (derin ağaçlar) en iyi çalışır.
:::

**Torba dışı (out-of-bag, OOB) hata:** her örnek, bootstrap örneklemlerinin ~%37'sinde **yoktur**. Her örneği yalnız onu görmemiş ağaçların ortalamasıyla tahmin etmek, ayrı bir doğrulama kümesi gerektirmeyen neredeyse yansız bir test hatası kestirimi verir.

Aşağıdaki deneyde formülü deneysel olarak doğrula: 24 farklı veri kümesinde bagging'in varyansını ölç, formülün tahminiyle karşılaştır; alt örneklemeyle ρ'nun nasıl düştüğünü gör.

::lab l2

### Rastgele orman: ρ'yu düşürmek {#rf}

Formül, varyansın tabanının $\rho\sigma^2$ olduğunu söylüyor. Ağaçları **daha az ilişkili** yapmak tabanı düşürür. Bagging'de güçlü bir öznitelik varsa her ağaç köke onu koyar ve ağaçlar birbirine benzer. **Rastgele orman** (Breiman 2001) her **bölünmede** özniteliklerin yalnız rastgele bir alt kümesine ($m\approx\sqrt d$ sınıflandırmada, $d/3$ regresyonda) bakar. Tek tek ağaçlar biraz kötüleşir ($\sigma^2$ ve yanlılık hafifçe artar) ama $\rho$ belirgin düşer; toplamda varyans azalır.

Rastgele orman, ayarlanması en az gereken güçlü yöntemlerden biridir: varsayılan ayarlarla bile çoğu tablo veri kümesinde iyi çalışır, ölçekleme istemez, OOB ile kendi doğrulamasını yapar ve permütasyon önemleriyle yorumlanabilir.

### Boosting: hataları sırayla düzeltmek {#boosting}

Boosting fikri (Schapire 1990, Freund ve Schapire 1997 AdaBoost): her biri rastgele tahminden yalnız biraz iyi olan **zayıf öğreniciler** sırayla eklenerek keyfi derecede iyi bir **güçlü öğrenici** elde edilebilir. AdaBoost bunu örnek ağırlıklarıyla yapar: her turda yanlış sınıflanan örneklerin ağırlığı artar, bir sonraki ağaç onlara odaklanır (M2.4 **e** aşamasındaki ağırlıklı bölünme bunun için). Sonradan anlaşıldı ki AdaBoost, üstel kaybı ileri aşamalı (stagewise) olarak küçültmektir (Friedman, Hastie, Tibshirani 2000). Bu bakış genel bir yönteme açıldı:

:::türetim Gradyan artırma = fonksiyon uzayında gradyan inişi
Amaç $\min_F\sum_iL(y_i,F(x_i))$, $F$ bir fonksiyon. Parametre uzayında gradyan inişi $\theta\leftarrow\theta-\eta\nabla_\theta L$ idi. Şimdi "parametre" $F$'nin eğitim noktalarındaki değerleri $F(x_1),\dots,F(x_n)$ olsun. Gradyan:
$$g_i=\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)},\qquad F(x_i)\leftarrow F(x_i)-\eta\,g_i.$$
Ama bu yalnız eğitim noktalarında tanımlı bir güncellemedir; yeni $x$'lere genelleşmez. Çözüm: negatif gradyanı ($r_i=-g_i$, **sözde artıklar**) bir **regresyon ağacıyla yaklaştır**, ağacı ekle:
$$F_0=\arg\min_c\sum_iL(y_i,c),\qquad h_m\approx\{(x_i,r_i^{(m)})\},\qquad F_m=F_{m-1}+\eta\,h_m.$$
**Kayba göre sözde artıklar:** MSE ($\frac12(y-F)^2$): $r=y-F$, sıradan artık. Mutlak hata: $r=\mathrm{sign}(y-F)$. Huber: kırpılmış artık. Lojistik kayıp: $r=y-\sigma(F)$ (M1.4'teki $p-y$'nin tersi). Kayıp değişince **yalnız gradyan hesabı** değişir — modülün **d** aşamasında bunu autograd'a bıraktık. (Tam algoritmada her yaprağın değeri ayrıca o yapraktaki kayba göre en iyi sabitle yeniden hesaplanır — "line search"; MSE'de bu yaprak ortalamasıdır.)
:::

**Shrinkage (öğrenme oranı η):** her ağacın katkısını küçültmek (0.01–0.1), aynı doğruluk için daha çok tur gerektirir ama genelleme neredeyse her zaman daha iyidir (Friedman 2001). Gradyan inişindeki küçük adımın karşılığı.

**Düzenlileştirme düğmeleri:** tur sayısı (erken durdurma ile), η, ağaç derinliği (etkileşim derecesi: derinlik $k$, en çok $k$'lı öznitelik etkileşimleri), yapraktaki en az örnek, satır/sütun alt örneklemesi (**stokastik gradyan artırma**: her turda verinin bir kısmı — varyansı düşürür, hızlandırır).

Aşağıda boosting'i adım adım izle: sözde artıkların nasıl küçüldüğünü, MAE'de nasıl işarete dönüştüğünü ve tur sayısıyla test hatasının nasıl önce düşüp sonra yükseldiğini gör.

::lab l1

:::türetim Regresyon stump'ının en iyi bölünmesi: S_L²/n_L + S_R²/n_R
Artıklar $r_i$, bir eşik solda $n_L$ örnek (toplamları $S_L$), sağda $n_R$ ($S_R$) bıraksın. Yapraklar ortalamayı tahmin eder: $\bar r_L=S_L/n_L$. Sol yaprağın SSE'si $\sum_{i\in L}(r_i-\bar r_L)^2=\sum_{i\in L}r_i^2-S_L^2/n_L$; sağ için aynı. Toplam SSE $=\sum_ir_i^2-\big(\frac{S_L^2}{n_L}+\frac{S_R^2}{n_R}\big)$. İlk terim bölünmeye bağlı değil: SSE'yi küçültmek $\frac{S_L^2}{n_L}+\frac{S_R^2}{n_R}$'yi büyütmektir. Sıralı artıkların kümülatif toplamı tüm eşiklerin $S_L$'sini tek geçişte verir (M2.4'teki hile, modülün **c** aşaması).
:::

### Modern gradyan artırma: XGBoost, LightGBM, CatBoost {#modern}

Tablo verisinde (satırlar örnek, sütunlar heterojen öznitelikler) gradyan artırmalı ağaçlar bugün de çoğu zaman derin ağlardan iyidir (Grinsztajn ve ark. 2022). Kütüphanelerin katkıları:

- **İkinci dereceden adım (XGBoost):** kaybın Taylor açılımında Hessian'ı da kullanır; yaprak değeri $-\frac{\sum g_i}{\sum h_i+\lambda}$, bölünme kazancı da buna göre. Yaprak sayısına ve yaprak değerlerine açık L2/L1 cezası.
- **Histogram tabanlı bölünme (LightGBM):** sürekli öznitelikleri 255 kutuya önceden ayırıp aday eşik sayısını sabitler: büyük veride çok hızlı. Yaprak öncelikli (leaf-wise) büyütme.
- **Kategorik öznitelikler (CatBoost):** hedef kodlamasını sızıntısız yapan sıralı (ordered) şemalar.
- Eksik değerleri öğrenilen bir varsayılan yönle işleme, GPU eğitimi, monotonluk kısıtları.

### Ne zaman hangisi? {#secim}

- **Hızlı, sağlam bir taban çizgisi:** rastgele orman (az ayar, paralel, OOB).
- **En iyi doğruluk, tablo verisi:** gradyan artırma (η, derinlik, tur sayısı ve alt örneklemeyi doğrulamayla ayarla; erken durdurma).
- **Görüntü, ses, metin:** derin ağlar; topluluklar burada da kullanılır (birkaç ağın ortalaması, "deep ensembles" — belirsizlik kestirimi için de, M3.9).

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| bagging | bootstrap + ortalama; $\mathrm{Var}=\rho\sigma^2+\frac{1-\rho}B\sigma^2$; yanlılık değişmez |
| OOB | her örneği onu görmeyen ağaçlarla tahmin et: bedava doğrulama |
| rastgele orman | bölünme başına öznitelik alt kümesi → $\rho$ düşer |
| boosting | sıralı; zayıf öğreniciler → güçlü öğrenici; yanlılığı düşürür |
| gradyan artırma | sözde artık $=-\partial L/\partial F$; her tura bir ağaç; $F_m=F_{m-1}+\eta h_m$ |
| düzenlileştirme | küçük η + çok tur + erken durdurma, sığ ağaçlar, alt örnekleme |

**Kaynaklar:** James ve ark., *ISLP* §8.2; Hastie ve ark., *ESL* Bölüm 10 (boosting, AdaBoost = üstel kayıp) ve 15 (rastgele orman, ρσ² formülü); Breiman (1996), "Bagging predictors"; Breiman (2001), "Random forests"; Freund ve Schapire (1997), AdaBoost; Friedman (2001), "Greedy function approximation: a gradient boosting machine"; Chen ve Guestrin (2016), XGBoost; Ke ve ark. (2017), LightGBM; Prokhorenkova ve ark. (2018), CatBoost; Grinsztajn, Oyallon, Varoquaux (2022), "Why do tree-based models still outperform deep learning on tabular data?".
