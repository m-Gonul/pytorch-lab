### Fikir: soru sorarak sınıflandırmak {#fikir}

"Yirmi soru" oyunu gibi: bir dizi evet/hayır sorusuyla olası cevapları daraltırsın. **Karar ağacı** her iç düğümde tek bir özniteliğe tek bir soru sorar — "$x_j\le t$ mi?" — ve cevaba göre sol ya da sağ dala gider. Bir yaprağa ulaşınca o yaprağın tahminini verir: sınıflandırmada yapraktaki çoğunluk sınıfı (ya da sınıf oranları), regresyonda yapraktaki hedeflerin ortalaması.

Geometride her soru uzayı bir eksene **paralel** bir düzlemle ikiye böler. Ağacın tamamı uzayı **eksene paralel dikdörtgenlere** (yüksek boyutta kutulara) ayırır ve her kutuda sabit bir tahmin verir: parçalı sabit bir fonksiyon.

<figure class="fig wide"><svg viewBox="0 0 640 230" role="img" aria-label="Solda üç iç düğümlü bir karar ağacı, sağda aynı ağacın girdi uzayında oluşturduğu dört dikdörtgen bölge"><g class="sm"><rect x="90" y="10" width="120" height="30" rx="6" class="mute" fill="none"/><text x="150" y="30" text-anchor="middle">x₁ ≤ 0.5 ?</text><line x1="120" y1="40" x2="70" y2="80" class="ink"/><line x1="180" y1="40" x2="230" y2="80" class="ink"/><text x="80" y="62">evet</text><text x="205" y="62">hayır</text><rect x="10" y="80" width="120" height="30" rx="6" class="mute" fill="none"/><text x="70" y="100" text-anchor="middle">x₂ ≤ 0.7 ?</text><rect x="170" y="80" width="120" height="30" rx="6" class="mute" fill="none"/><text x="230" y="100" text-anchor="middle">x₂ ≤ 0.3 ?</text><line x1="45" y1="110" x2="30" y2="150" class="ink"/><line x1="95" y1="110" x2="110" y2="150" class="ink"/><line x1="205" y1="110" x2="190" y2="150" class="ink"/><line x1="255" y1="110" x2="270" y2="150" class="ink"/><rect x="5" y="150" width="50" height="26" rx="5" class="soft-a"/><text x="30" y="167" text-anchor="middle">A</text><rect x="85" y="150" width="50" height="26" rx="5" class="soft-b"/><text x="110" y="167" text-anchor="middle">B</text><rect x="165" y="150" width="50" height="26" rx="5" class="soft-b"/><text x="190" y="167" text-anchor="middle">C</text><rect x="245" y="150" width="50" height="26" rx="5" class="soft-a"/><text x="270" y="167" text-anchor="middle">D</text><text x="150" y="205" text-anchor="middle">iç düğüm: soru · yaprak: tahmin</text></g><g transform="translate(360,10)" class="sm"><rect x="0" y="0" width="100" height="60" class="soft-b"/><rect x="0" y="60" width="100" height="140" class="soft-a"/><rect x="100" y="0" width="160" height="140" class="soft-a"/><rect x="100" y="140" width="160" height="60" class="soft-b"/><rect x="0" y="0" width="260" height="200" class="ink" fill="none"/><line x1="100" y1="0" x2="100" y2="200" class="ink w2"/><line x1="0" y1="60" x2="100" y2="60" class="ink w2"/><line x1="100" y1="140" x2="260" y2="140" class="ink w2"/><text x="50" y="135" text-anchor="middle">A</text><text x="50" y="35" text-anchor="middle">B</text><text x="180" y="175" text-anchor="middle">C</text><text x="180" y="75" text-anchor="middle">D</text><text x="100" y="215" text-anchor="middle">x₁ = 0.5</text><text x="4" y="56">x₂ = 0.7</text><text x="256" y="136" text-anchor="end">x₂ = 0.3</text></g></svg><figcaption>Şekil 1 — Solda ağaç, sağda aynı ağacın girdi uzayını bölüşü (x₂ ekseni yukarı doğru artar; alt kenar x₂ = 0). Her yol bir dikdörtgene, her yaprak bir sabit tahmine karşılık gelir.</figcaption></figure>

Ağaçlar gradyan kullanmaz: öğrenme, verinin **açgözlü** biçimde bölünmesidir. Bu, M1'deki "kayıp + gradyan inişi" kalıbının dışında kalan ilk modelimiz — ve tablo verisinde hâlâ en güçlü yöntemlerin (M2.5: rastgele orman, gradyan artırma) yapı taşı.

### Safsızlık: bir düğüm ne kadar "karışık"? {#safsizlik}

Bir düğümdeki örneklerin sınıf oranları $p_1,\dots,p_K$ olsun. İyi bir bölünme, çocukları ebeveynden daha **saf** (tek sınıfa yakın) yapar. Üç ölçüt:

| ölçüt | formül | yorum |
|---|---|---|
| Gini | $G=1-\sum_kp_k^2=\sum_kp_k(1-p_k)$ | rastgele seçilen bir örneğe, düğümün dağılımından rastgele etiket verilirse yanlış olma olasılığı |
| entropi | $H=-\sum_kp_k\log_2p_k$ | etiketi kodlamak için gereken ortalama bit (M0.4) |
| sınıflandırma hatası | $E=1-\max_kp_k$ | çoğunluk etiketiyle yapılan hata oranı |

<figure class="fig"><svg viewBox="0 0 420 240" role="img" aria-label="İkili sınıflandırmada Gini, entropi ve sınıflandırma hatası, p'ye göre"><line x1="40" y1="200" x2="390" y2="200" class="mute"/><line x1="40" y1="200" x2="40" y2="20" class="mute"/><polyline points="40.0,200.0 43.4,186.3 46.8,176.0 50.2,167.0 53.6,158.8 57.0,151.3 60.4,144.3 63.8,137.8 67.2,131.6 70.6,125.8 74.0,120.3 77.4,115.0 80.8,110.0 84.2,105.2 87.6,100.7 91.0,96.3 94.4,92.2 97.8,88.2 101.2,84.4 104.6,80.7 108.0,77.3 111.4,73.9 114.8,70.8 118.2,67.7 121.6,64.8 125.0,62.1 128.4,59.5 131.8,57.0 135.2,54.6 138.6,52.3 142.0,50.2 145.4,48.2 148.8,46.3 152.2,44.5 155.6,42.8 159.0,41.2 162.4,39.7 165.8,38.4 169.2,37.1 172.6,36.0 176.0,34.9 179.4,34.0 182.8,33.2 186.2,32.4 189.6,31.8 193.0,31.2 196.4,30.8 199.8,30.4 203.2,30.2 206.6,30.0 210.0,30.0 213.4,30.0 216.8,30.2 220.2,30.4 223.6,30.8 227.0,31.2 230.4,31.8 233.8,32.4 237.2,33.2 240.6,34.0 244.0,34.9 247.4,36.0 250.8,37.1 254.2,38.4 257.6,39.7 261.0,41.2 264.4,42.8 267.8,44.5 271.2,46.3 274.6,48.2 278.0,50.2 281.4,52.3 284.8,54.6 288.2,57.0 291.6,59.5 295.0,62.1 298.4,64.8 301.8,67.7 305.2,70.8 308.6,73.9 312.0,77.3 315.4,80.7 318.8,84.4 322.2,88.2 325.6,92.2 329.0,96.3 332.4,100.7 335.8,105.2 339.2,110.0 342.6,115.0 346.0,120.3 349.4,125.8 352.8,131.6 356.2,137.8 359.6,144.3 363.0,151.3 366.4,158.8 369.8,167.0 373.2,176.0 376.6,186.3 380.0,200.0" class="cq w2" fill="none"/><polyline points="40.0,200.0 43.4,196.6 46.8,193.3 50.2,190.1 53.6,186.9 57.0,183.8 60.4,180.8 63.8,177.9 67.2,175.0 70.6,172.2 74.0,169.4 77.4,166.7 80.8,164.1 84.2,161.5 87.6,159.1 91.0,156.7 94.4,154.3 97.8,152.0 101.2,149.8 104.6,147.7 108.0,145.6 111.4,143.6 114.8,141.7 118.2,139.8 121.6,138.0 125.0,136.2 128.4,134.6 131.8,133.0 135.2,131.5 138.6,130.0 142.0,128.6 145.4,127.3 148.8,126.0 152.2,124.8 155.6,123.7 159.0,122.7 162.4,121.7 165.8,120.7 169.2,119.9 172.6,119.1 176.0,118.4 179.4,117.8 182.8,117.2 186.2,116.7 189.6,116.2 193.0,115.8 196.4,115.5 199.8,115.3 203.2,115.1 206.6,115.0 210.0,115.0 213.4,115.0 216.8,115.1 220.2,115.3 223.6,115.5 227.0,115.8 230.4,116.2 233.8,116.7 237.2,117.2 240.6,117.8 244.0,118.4 247.4,119.1 250.8,119.9 254.2,120.7 257.6,121.7 261.0,122.7 264.4,123.7 267.8,124.8 271.2,126.0 274.6,127.3 278.0,128.6 281.4,130.0 284.8,131.5 288.2,133.0 291.6,134.6 295.0,136.2 298.4,138.0 301.8,139.8 305.2,141.7 308.6,143.6 312.0,145.6 315.4,147.7 318.8,149.8 322.2,152.0 325.6,154.3 329.0,156.7 332.4,159.1 335.8,161.5 339.2,164.1 342.6,166.7 346.0,169.4 349.4,172.2 352.8,175.0 356.2,177.9 359.6,180.8 363.0,183.8 366.4,186.9 369.8,190.1 373.2,193.3 376.6,196.6 380.0,200.0" class="ca w2" fill="none"/><polyline points="40.0,200.0 43.4,198.3 46.8,196.6 50.2,194.9 53.6,193.2 57.0,191.5 60.4,189.8 63.8,188.1 67.2,186.4 70.6,184.7 74.0,183.0 77.4,181.3 80.8,179.6 84.2,177.9 87.6,176.2 91.0,174.5 94.4,172.8 97.8,171.1 101.2,169.4 104.6,167.7 108.0,166.0 111.4,164.3 114.8,162.6 118.2,160.9 121.6,159.2 125.0,157.5 128.4,155.8 131.8,154.1 135.2,152.4 138.6,150.7 142.0,149.0 145.4,147.3 148.8,145.6 152.2,143.9 155.6,142.2 159.0,140.5 162.4,138.8 165.8,137.1 169.2,135.4 172.6,133.7 176.0,132.0 179.4,130.3 182.8,128.6 186.2,126.9 189.6,125.2 193.0,123.5 196.4,121.8 199.8,120.1 203.2,118.4 206.6,116.7 210.0,115.0 213.4,116.7 216.8,118.4 220.2,120.1 223.6,121.8 227.0,123.5 230.4,125.2 233.8,126.9 237.2,128.6 240.6,130.3 244.0,132.0 247.4,133.7 250.8,135.4 254.2,137.1 257.6,138.8 261.0,140.5 264.4,142.2 267.8,143.9 271.2,145.6 274.6,147.3 278.0,149.0 281.4,150.7 284.8,152.4 288.2,154.1 291.6,155.8 295.0,157.5 298.4,159.2 301.8,160.9 305.2,162.6 308.6,164.3 312.0,166.0 315.4,167.7 318.8,169.4 322.2,171.1 325.6,172.8 329.0,174.5 332.4,176.2 335.8,177.9 339.2,179.6 342.6,181.3 346.0,183.0 349.4,184.7 352.8,186.4 356.2,188.1 359.6,189.8 363.0,191.5 366.4,193.2 369.8,194.9 373.2,196.6 376.6,198.3 380.0,200.0" class="cb w2" fill="none"/><text x="214" y="26" class="sm tq">entropi (bit)</text><text x="214" y="109" class="sm ta">Gini 1 − Σp²</text><text x="285" y="168" class="sm tb">hata min(p, 1 − p)</text><text x="40" y="216" class="sm" text-anchor="middle">0</text><text x="210" y="216" class="sm" text-anchor="middle">0.5</text><text x="380" y="216" class="sm" text-anchor="middle">1</text><text x="215" y="234" class="sm" text-anchor="middle">p = düğümdeki sınıf-1 oranı</text><text x="34" y="34" class="sm" text-anchor="end">1</text><text x="34" y="119" class="sm" text-anchor="end">0.5</text></svg><figcaption>Şekil 2 — Üç safsızlık ölçütü saf düğümlerde (p = 0 ya da 1) sıfır, en karışık düğümde (p = ½) en büyüktür. Gini ve entropi kesin içbükeydir; sınıflandırma hatası parçalı doğrusaldır.</figcaption></figure>

Bir bölünmenin skoru, çocukların **örnek sayısıyla ağırlıklandırılmış** safsızlığıdır; **kazanç** ebeveynle farkı:
$$\text{skor}(j,t)=\frac{n_L}{n}I(L)+\frac{n_R}{n}I(R),\qquad \Delta I=I(\text{ebeveyn})-\text{skor}(j,t)\ge0.$$
Entropiyle kazanca **bilgi kazancı** (information gain) denir: bölünmenin etiket hakkında verdiği karşılıklı bilgi.

:::türetim Kazanç neden hiç negatif olmaz ve neden sınıflandırma hatası kötü bir ölçüttür?
Ebeveynin sınıf oranı, çocuklarınkinin ağırlıklı ortalamasıdır: $p=\frac{n_L}np_L+\frac{n_R}np_R$. $I$ içbükey olduğu için (Jensen) $I(p)\ge\frac{n_L}nI(p_L)+\frac{n_R}nI(p_R)$: kazanç $\ge0$. Eşitlik, $I$ $p_L$ ile $p_R$ arasında **doğrusalsa** olur. Gini ve entropi kesin içbükeydir: $p_L\ne p_R$ olan her bölünme pozitif kazanç verir. Sınıflandırma hatası ise parçalı doğrusaldır: örneğin $p=0.7$'lik bir düğümü $p_L=0.6$, $p_R=0.9$ (ikisi de çoğunluk 1) diye bölmek hatayı hiç değiştirmez, oysa sağ çocuk açıkça daha saftır ve sonraki bölünmeler için daha iyi bir başlangıçtır. Bu yüzden ağaç büyütmede Gini ya da entropi, budamada ise sınıflandırma hatası kullanılır.
:::

**Gini mi entropi mi?** Pratikte neredeyse her zaman aynı bölünmeyi seçerler; Gini logaritma istemediği için biraz daha hızlıdır (scikit-learn varsayılanı).

**Regresyon ağaçları:** safsızlık **varyanstır** (MSE), yaprak tahmini ortalamadır; bölünme skoru $\frac{n_L}n\mathrm{Var}(y_L)+\frac{n_R}n\mathrm{Var}(y_R)$. Gradyan artırma (M2.5) regresyon ağaçlarını kullanır.

### Açgözlü ağaç kurma (CART) {#cart}

En iyi ağacı bulmak (en az yapraklı, en az hatalı) NP-zordur (Hyafil ve Rivest 1976). CART (Breiman ve ark. 1984) yerine açgözlü bir yöntem kullanır:

1. Düğümdeki örnekler için tüm öznitelikleri $j$ ve tüm aday eşikleri $t$ dene; skoru en küçük $(j,t)$'yi seç.
2. Durdurma koşulu sağlanmıyorsa veriyi böl ve iki çocukta aynı şeyi özyinelemeli yap.
3. Durunca yaprak yap; tahmini yapraktaki örneklerden hesapla.

**Aday eşikler:** bir öznitelikte yalnız örnekler arası geçişler önemlidir: sıralı farklı değerlerin **orta noktaları** ($n-1$ aday). Her adayın skorunu ayrı ayrı hesaplamak $O(n^2)$'dir; özniteliği bir kez sıralayıp sınıf sayımlarını **kümülatif** toplayınca tüm adaylar $O(n)$'de değerlendirilir: sıralı sırayla $i$. eşikte sol çocukta ilk $i$ örnek vardır, sol sınıf sayımları one-hot etiketlerin kümülatif toplamının $i$. satırıdır (modülün **c** aşaması, döngüsüz). Bir düğümün toplam maliyeti $O(dn\log n)$, tüm ağaç kabaca $O(dn\log^2n)$.

**Durdurma ölçütleri:** en büyük derinlik, yaprakta en az örnek, bölünmede en az kazanç, düğüm saflığı. Bunlar düzenlileştirmedir: ağacın karmaşıklığını sınırlarlar.

### Aşırı uyum ve budama {#budama}

Sınırsız büyüyen bir ağaç her örneği kendi yaprağına koyar: eğitim doğruluğu %100, ama yapraklar gürültüyü ezberler. Derinlik yanlılık–varyans takasının düğmesidir: sığ ağaç yanlı (eğik bir sınırı kaba bir merdivenle yaklaştırır), derin ağaç yüksek varyanslı.

**Maliyet–karmaşıklık budaması** (CART'ın yöntemi): önce büyük bir ağaç büyüt, sonra
$$R_\alpha(T)=R(T)+\alpha\,|T|$$
($R$: eğitim hatası, $|T|$: yaprak sayısı) en küçük olan alt ağacı seç. $\alpha$ arttıkça budanmış ağaçlar iç içe bir dizi oluşturur ("en zayıf bağlantı" budaması); $\alpha$ çapraz doğrulamayla seçilir. Önce büyütüp sonra budamak, erken durdurmaktan iyidir: XOR gibi problemlerde ilk bölünme kazançsız görünür ama sonraki bölünmeler çok değerlidir — erken durdurma bunu kaçırır.

Aşağıda veri kümesini, derinliği ve yaprak boyunu değiştir; dikdörtgen bölgeleri ve eğitim/test doğruluğunun derinlikle ayrışmasını gör.

::lab l1

### Ağaçların karakteri {#karakter}

| güçlü yan | zayıf yan |
|---|---|
| yorumlanabilir: küçük bir ağaç bir kurallar listesidir | **kararsız**: veride küçük bir değişiklik kökü ve tüm ağacı değiştirebilir (yüksek varyans) |
| ölçek gerektirmez: tekdüze dönüşümlere değişmez (sıralama korunur) | sınırlar eksene paralel: eğik ya da pürüzsüz sınırlar için çok yaprak gerekir |
| karışık öznitelik türleri, etkileşimleri kendiliğinden yakalar | açgözlü: ileriyi göremez (XOR'da ilk bölünme kazançsız) |
| eksik değerlerle başa çıkma yolları var (vekil bölünmeler) | tahmin parçalı sabit: regresyonda pürüzlü, eğitim aralığı dışına çıkamaz |
| çıkarım çok hızlı: derinlik kadar karşılaştırma | tek başına doğruluğu genelde orta |

Kararsızlık ve yüksek varyans, M2.5'teki **topluluk** yöntemlerinin hareket noktasıdır: birçok ağacın ortalaması (bagging, rastgele orman) varyansı düşürür; ağaçları sırayla birbirinin hatasını düzeltecek biçimde eklemek (boosting) yanlılığı düşürür.

**Öznitelik önemi:** bir özniteliğin, kullanıldığı tüm bölünmelerdeki (örnek sayısıyla ağırlıklı) safsızlık azalışlarının toplamı. Hızlıdır ama yanlıdır: çok sayıda farklı değeri olan öznitelikler (kimlik numarası gibi) çok aday eşik sunduğu için şansla yüksek kazanç bulur. Daha güvenilir alternatif: **permütasyon önemi** (bir özniteliğin değerlerini karıştırınca doğrulama hatası ne kadar artıyor).

### Yumuşak ağaçlar: türevlenebilir köprü {#yumusak}

Sert soru $[x_j\le t]$ basamak fonksiyonudur: $t$'ye göre türevi her yerde 0. Onu yumuşak bir kapıyla değiştir: sağa gitme olasılığı $\sigma\big((w^\top x+b)/\tau\big)$. Her örnek her yaprağa, yol üzerindeki kapı olasılıklarının çarpımı kadar ağırlıkla gider; tahmin yaprak değerlerinin ağırlıklı toplamıdır. Model artık türevlenebilir ve autograd ile eğitilebilir (modülün **d** aşaması). $\tau\to0$'da kapılar sertleşir ve sıradan bir ağaç (ama eğik bölünmeli) elde edilir. Yumuşak ağaçlar, ağaçların yapısını sinir ağlarının eğitim araçlarıyla birleştiren bir araştırma çizgisidir (Frosst ve Hinton 2017; neural oblivious decision ensembles).

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| model | eksene paralel kutular, kutu başına sabit tahmin |
| safsızlık | Gini $1-\sum p_k^2$, entropi $-\sum p_k\log p_k$; regresyonda varyans |
| bölünme skoru | çocukların ağırlıklı safsızlığı; kazanç içbükeylik sayesinde $\ge0$ |
| CART | açgözlü; aday eşikler orta noktalar; sırala + kümülatif sayım |
| aşırı uyum | derinlik ve yaprak boyu; büyüt sonra buda ($R+\alpha|T|$) |
| karakter | ölçeksiz, yorumlanabilir, kararsız → topluluklar (M2.5) |

**Kaynaklar:** Breiman, Friedman, Olshen, Stone (1984), *Classification and Regression Trees*; James ve ark., *ISLP* Bölüm 8.1; Hastie ve ark., *ESL* §9.2 (safsızlık ölçütleri, maliyet–karmaşıklık budaması); scikit-learn kullanıcı kılavuzu, "Decision trees"; Hyafil ve Rivest (1976), en iyi ikili karar ağacının NP-tamlığı; Strobl ve ark. (2007), safsızlık tabanlı öznitelik öneminin yanlılığı; Frosst ve Hinton (2017), "Distilling a neural network into a soft decision tree".
