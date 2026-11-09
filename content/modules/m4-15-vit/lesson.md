### Görüntüde dikkat: altyazı üretiminden ViT'e {#gorsel}

Dikkat mekanizmaları bugün çok çeşitli amaçlarla kullanılıyor. Makine çevirisinden sonraki ilk uygulamalarından biri **görsel dikkatle görüntü altyazısı** üretmekti (Xu ve ark. 2015, "Show, Attend and Tell"). Bir CNN önce görüntüyü işleyip öznitelik haritaları çıkarır. Sonra dikkat mekanizmalı bir RNN kod çözücü altyazıyı kelime kelime üretir. Bu, M4.13'teki Bahdanau modelinin aynısıdır; tek fark, "kaynak cümle" yerine **uzamsal konumların** bulunmasıdır.

:::tanım Görsel dikkat (Xu ve ark. 2015)
CNN'in bir ara katmanı $14\times14\times512$'lik bir öznitelik haritası verir. Bunu $L=196$ konumun her biri için 512 boyutlu bir **not** (annotation) $\mathbf a_i$ olarak okuruz. Her kelime adımında kod çözücünün durumu $\mathbf h_{t-1}$ ile her not arasında bir skor $e_{ti}=f_{\text{att}}(\mathbf a_i,\mathbf h_{t-1})$ hesaplanır (toplamsal, M4.13). Softmax ağırlıkları $\alpha_{ti}$ ve bağlam $\hat{\mathbf z}_t=\sum_i\alpha_{ti}\mathbf a_i$ bulunur; LSTM bu bağlamı girdi olarak alır. Bu, **yumuşak** (soft) dikkattir ve türevlenebilir. Makale ayrıca tek bir konumu örnekleyen **sert** (hard) dikkati de dener; sert dikkat türevlenemez ve pekiştirmeli öğrenmeyle (REINFORCE, Faz 11) eğitilir.
:::

Her kod çözücü adımında (her kelimede) model dikkat modeliyle görüntünün yalnız ilgili kısmına odaklanır. Ağırlıklar görüntünün üzerine yukarı örneklenip çizilince bir **dikkat haritası** elde edilir; yüksek ağırlıklı pikseller daha parlaktır. Slayttaki örnekte model "A woman is throwing a frisbee in a park" altyazısını üretmiş; "frisbee" kelimesini üretmek üzereyken dikkatinin büyük kısmı frizbinin üzerindedir. (Şekil görüntü olduğu için metne dökülemedi.) Bu haritalar yorumlamaya yardım eder, ama M4.13'te Jain ve Wallace'tan aktarılan uyarı burada da geçerlidir: dikkat ağırlığı nedenselliğin kanıtı değildir.

Bu düzende görüntüyü hâlâ bir CNN işler; dikkat yalnız kod çözücüdedir. Sonraki adım soruyu tersine çevirdi: CNN'e hiç gerek var mı? Transformer mimarisi görme problemlerine de uygulanabilir mi? 2020'de Google'ın "An Image is Worth 16x16 Words" makalesi (Dosovitskiy ve ark., ICLR 2021) bu soruya evet dedi: **Vision Transformer** (ViT).

### ViT: görüntüyü kelimelere bölmek {#vit}

Transformer bir token dizisi bekler. Her pikseli bir token yapmak mümkün değildir: $224\times224$'lük bir görüntü 50 176 token olur ve öz-dikkatin $O(n^2)$ maliyeti (M4.14) $2.5\times10^9$ skor demektir. Daha önce yerel dikkat, seyrek dikkat ve $2\times2$'lik yamalar gibi çözümler denenmişti (Cordonnier ve ark. 2020'nin $2\times2$ yamaları yalnız küçük çözünürlüklere uygulanabiliyordu). ViT'in fikri basittir: görüntüyü **$P\times P$'lik yamalara** böl ve her yamayı bir kelime gibi bir token yap.

:::tanım ViT (Dosovitskiy ve ark. 2021)
Görüntü $\mathbf x\in\mathbb R^{H\times W\times C}$, $N=HW/P^2$ adet düzleştirilmiş yamaya bölünür: $\mathbf x_p\in\mathbb R^{N\times(P^2C)}$. Her yama öğrenilen bir doğrusal izdüşümle ($\mathbf E\in\mathbb R^{(P^2C)\times D}$) $D$ boyutlu bir **yama embedding'ine** çevrilir. Başa öğrenilebilir bir **[class] tokenı** eklenir (BERT'in [CLS]'i gibi, M4.16). Öğrenilebilir 1-B **konum embedding'leri** $\mathbf E_{pos}\in\mathbb R^{(N+1)\times D}$ toplanır:
$$\mathbf z_0=[\mathbf x_{\text{class}};\ \mathbf x_p^1\mathbf E;\ \dots;\ \mathbf x_p^N\mathbf E]+\mathbf E_{pos},$$
$$\mathbf z'_\ell=\mathrm{MSA}(\mathrm{LN}(\mathbf z_{\ell-1}))+\mathbf z_{\ell-1},\qquad\mathbf z_\ell=\mathrm{MLP}(\mathrm{LN}(\mathbf z'_\ell))+\mathbf z'_\ell,\qquad\ell=1..L,$$
$$\mathbf y=\mathrm{LN}(\mathbf z_L^0).$$
$\mathbf z_L^0$, son katmandaki [class] tokenının durumudur. Üstüne ön eğitimde tek gizli katmanlı bir MLP, ince ayarda tek bir doğrusal katman olan sınıflandırma başı konur. MSA: çok başlı öz-dikkat (M4.14). MLP: iki katman, GELU aktivasyon. LayerNorm her bloktan **önce** gelir (Pre-LN, M4.7).
:::

<figure class="fig wide"><svg viewBox="0 0 640 450" role="img" aria-label="Vision Transformer: görüntü yamalara bölünür, her yama düzleştirilip doğrusal olarak D boyuta izdüşürülür, başa sınıf tokenı eklenir, konum embedding'leri toplanır, L katmanlı Pre-LN Transformer kodlayıcıdan geçer ve sınıf tokenının çıktısı sınıflandırma başına verilir"><defs><marker id="v1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="20" y="40" width="21" height="21" class="soft-p"/><rect x="42" y="40" width="21" height="21" class="soft-a"/><rect x="64" y="40" width="21" height="21" class="soft-p"/><rect x="86" y="40" width="21" height="21" class="soft-a"/><rect x="20" y="62" width="21" height="21" class="soft-a"/><rect x="42" y="62" width="21" height="21" class="soft-p"/><rect x="64" y="62" width="21" height="21" class="soft-a"/><rect x="86" y="62" width="21" height="21" class="soft-p"/><rect x="20" y="84" width="21" height="21" class="soft-p"/><rect x="42" y="84" width="21" height="21" class="soft-a"/><rect x="64" y="84" width="21" height="21" class="soft-p"/><rect x="86" y="84" width="21" height="21" class="soft-a"/><rect x="20" y="106" width="21" height="21" class="soft-a"/><rect x="42" y="106" width="21" height="21" class="soft-p"/><rect x="64" y="106" width="21" height="21" class="soft-a"/><rect x="86" y="106" width="21" height="21" class="soft-p"/><text x="20" y="30" class="sm">görüntü H × W × C</text><text x="20" y="144" class="sm">N = HW/P² yama</text><line x1="114" y1="84" x2="130" y2="84" class="ink" marker-end="url(#v1)"/><rect x="134" y="70" width="15" height="30" class="soft-p"/><rect x="152" y="70" width="15" height="30" class="soft-a"/><rect x="170" y="70" width="15" height="30" class="soft-p"/><rect x="188" y="70" width="15" height="30" class="soft-a"/><rect x="206" y="70" width="15" height="30" class="soft-p"/><rect x="224" y="70" width="15" height="30" class="soft-a"/><text x="134" y="60" class="sm">düzleştir: P²C</text><text x="240" y="90" class="sm">…</text><line x1="190" y1="104" x2="190" y2="124" class="ink" marker-end="url(#v1)"/><rect x="130" y="126" width="130" height="26" rx="5" class="soft-b"/><text x="195.0" y="143.0" text-anchor="middle" class="sm">doğrusal izdüşüm E</text><text x="130" y="168" class="sm">= Conv2d(C, D, P, adım P)</text><line x1="260" y1="139" x2="290" y2="139" class="ink" marker-end="url(#v1)"/><rect x="294" y="126" width="48" height="26" rx="5" class="soft-p"/><text x="318.0" y="143.0" text-anchor="middle" class="sm">[class]</text><rect x="346" y="126" width="48" height="26" rx="5" class="soft-a"/><text x="370.0" y="143.0" text-anchor="middle" class="sm">x¹E</text><rect x="398" y="126" width="48" height="26" rx="5" class="soft-a"/><text x="422.0" y="143.0" text-anchor="middle" class="sm">x²E</text><rect x="450" y="126" width="48" height="26" rx="5" class="soft-a"/><text x="474.0" y="143.0" text-anchor="middle" class="sm">…</text><rect x="502" y="126" width="48" height="26" rx="5" class="soft-a"/><text x="526.0" y="143.0" text-anchor="middle" class="sm">xᴺE</text><text x="294" y="170" class="sm">+ E<tspan baseline-shift="-3" font-size="75%">pos</tspan>: her konuma öğrenilen bir vektör</text><line x1="450" y1="176" x2="450" y2="196" class="ink" marker-end="url(#v1)"/><rect x="363" y="200" width="190" height="160" rx="8" class="mute dash"/><text x="547" y="284" class="sm">L ×</text><rect x="375" y="210" width="150" height="24" rx="5" class="soft-p"/><text x="450.0" y="226.0" text-anchor="middle" class="sm">LayerNorm</text><line x1="450" y1="234" x2="450" y2="246" class="ink" marker-end="url(#v1)"/><rect x="375" y="246" width="150" height="24" rx="5" class="soft-b"/><text x="450.0" y="262.0" text-anchor="middle" class="sm">çok başlı öz-dikkat</text><line x1="450" y1="270" x2="450" y2="282" class="ink" marker-end="url(#v1)"/><rect x="375" y="282" width="150" height="24" rx="5" class="soft-p"/><text x="450.0" y="298.0" text-anchor="middle" class="sm">LayerNorm</text><line x1="450" y1="306" x2="450" y2="318" class="ink" marker-end="url(#v1)"/><rect x="375" y="318" width="150" height="24" rx="5" class="soft-b"/><text x="450.0" y="334.0" text-anchor="middle" class="sm">MLP (GELU, 4D)</text><text x="531" y="266" class="sm">+ artık</text><text x="531" y="338" class="sm">+ artık</text><line x1="450" y1="360" x2="450" y2="376" class="ink" marker-end="url(#v1)"/><rect x="365" y="378" width="170" height="26" rx="5" class="soft-p"/><text x="450.0" y="395.0" text-anchor="middle" class="sm">LN(z<tspan baseline-shift="-3" font-size="75%">L</tspan>[class]) → MLP başı</text><line x1="450" y1="404" x2="450" y2="418" class="ink" marker-end="url(#v1)"/><rect x="390" y="420" width="120" height="24" rx="5" class="soft-a"/><text x="450.0" y="436.0" text-anchor="middle" class="sm">sınıf olasılıkları</text><text x="20" y="260" class="sm">Pre-LN blok:</text><text x="20" y="276" class="sm">z′ = MSA(LN(z)) + z</text><text x="20" y="292" class="sm">z = MLP(LN(z′)) + z′</text><text x="20" y="324" class="sm">ViT-B/16: D = 768, L = 12,</text><text x="20" y="340" class="sm">12 baş, MLP 3072, 86M parametre.</text></svg><figcaption>Şekil 1 — ViT'in akışı (Dosovitskiy ve ark. 2021). Görüntüye özgü tek işlem yamalara bölmektir; geri kalanı NLP'deki Transformer kodlayıcısıyla (M4.14) neredeyse aynıdır. Farklar: LayerNorm her alt katmandan önce gelir (Pre-LN), MLP'de GELU kullanılır ve kod çözücü yoktur.</figcaption></figure>

Model adlandırması: "ViT-L/16", 16 × 16 yamalı Large modeldir. Dizi uzunluğu yama boyunun karesiyle **ters** orantılıdır, bu yüzden küçük yamalı modeller hesapça daha pahalıdır.

| model | katman $L$ | $D$ | MLP | baş | parametre |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

:::örnek El hesabı: yamalar ve dizi uzunluğu
$224\times224$, $P=16$: $N=(224/16)^2=14^2=\mathbf{196}$ yama; [class] ile dizi uzunluğu **197**. Her yama $16\cdot16\cdot3=768$ sayıdır. ViT-B'de $D=768$ olduğu için izdüşüm matrisi kare çıkar; bu bir tesadüftür. 384 × 384'te ince ayar: $N=24^2=\mathbf{576}$, dizi 577. Çözünürlük $384/224=1.71$ kat artarken dikkat skor matrisi $(577/197)^2=\mathbf{8.58}$ kat büyür.
:::

:::türetim Yama embedding'i = adımı $P$ olan konvolüsyon
Yama $\mathbf x_p^n$'yi $(p_1,p_2,c)$ sırasıyla düzleştirip $\mathbf E$ ile çarpmak, $d$. çıktı bileşeni için
$$(\mathbf x_p^n\mathbf E)_d=\sum_{p_1,p_2,c}x[c,\ P\,r_n+p_1,\ P\,s_n+p_2]\,E[(p_1,p_2,c),d]$$
demektir; $(r_n,s_n)$ yamanın ızgaradaki satır ve sütunudur. Çekirdeği $K[d,c,p_1,p_2]=E[(p_1,p_2,c),d]$ olan, $P\times P$ boyutlu, **adımı $P$** olan bir konvolüsyonun $(r_n,s_n)$ konumundaki çıktısı tam olarak budur. Adım çekirdek boyuna eşit olduğu için pencereler örtüşmez ve görüntüyü tam kaplar. Bu yüzden PyTorch uygulamaları yama embedding'ini `nn.Conv2d(C, D, kernel_size=P, stride=P)` olarak yazar. Aşama c'de `conv_to_linear` ile iki biçimin sayısal eşitliği doğrulanıyor. Ağırlıkların sırası (önce $c$ mi, $p$ mi) yalnız bir yeniden düzenleme meselesidir.
:::

:::türetim ViT-B/16'nın parametre sayımı
$D=768$, $L=12$, MLP 3072, 1000 sınıf:
- Yama embedding'i: $16\cdot16\cdot3\cdot768+768=\mathbf{590\,592}$.
- [class] tokenı: 768. Konum embedding'i: $197\cdot768=151\,296$.
- Bir katman: MSA $4(768^2+768)=2\,362\,368$; MLP $768\cdot3072+3072+3072\cdot768+768=4\,722\,432$; iki LN $4\cdot768=3\,072$. Toplam $\mathbf{7\,087\,872}$.
- Son LN: $1\,536$. Doğrusal baş: $768\cdot1000+1000=769\,000$.

**Toplam:** $590\,592+768+151\,296+12\cdot7\,087\,872+1\,536+769\,000=\mathbf{86\,567\,656}\approx86$M. Parametrelerin %98'i kodlayıcı katmanlarındadır. Görüntüye özgü kısım (yama embedding'i + konum) yalnız %0.9'dur.
:::

Aşağıdaki deneyde görüntü boyunu ve yama boyunu değiştir; dizi uzunluğunun ve dikkat/MLP hesap oranının ($T/4D$) nasıl değiştiğini, konum bilgisi olmadan modelin gördüğü "yama torbasını" incele.

::lab l1

### Konum embedding'leri ve [class] tokenı {#konum}

M4.14'te gösterildiği gibi öz-dikkat permütasyona eş-değişkendir. Konum embedding'i olmayan bir ViT, yamaların hangi sırada geldiğini bilmez: görüntü onun için bir **yama torbasıdır** (bag of patches). Ek x2 bunu sayısal olarak doğruluyor: yamaları karıştırınca [class] çıktısı değişmez.

ViT standart, öğrenilebilir **1-B** konum embedding'leri kullanır: yamalar satır satır (raster sırası) numaralanır ve her numaraya bir vektör öğrenilir. Yazarlar 2-B farkındalıklı sürümleri de denedi (Ek D.4) ama anlamlı bir kazanç görmedi. ViT-B/16, ImageNet 5-shot doğrusal değerlendirme:

| konum bilgisi | doğruluk |
|---|---|
| yok (yama torbası) | 0.614 |
| 1-B, öğrenilen | **0.642** |
| 2-B (satır ve sütun için ayrı $D/2$ boyutlu embedding'ler) | 0.640 |
| göreli (1-B göreli dikkat) | 0.640 |

Konumsuz model 2.8 puan geridedir; konum bilgisinin **nasıl** kodlandığı ise neredeyse fark etmez. Yazarların açıklaması: yama düzeyinde (14 × 14) uzamsal çözünürlük düşüktür ve 1-B embedding'ler bile 2-B yapıyı öğrenebilir. Makalenin Şekil 7'si öğrenilmiş embedding'lerin birbirine benzerliğini çizer: yakın yamaların embedding'leri benzerdir, aynı satır ve sütundakiler de benzerdir. Model 1-B indislerden 2-B ızgarayı **kendiliğinden** keşfetmiştir.

**[class] tokenı mı, ortalama mı?** Tüm yama tokenlarının ortalaması (global average pooling, GAP; CNN'lerde olduğu gibi, M4.8) da kullanılabilir. Yazarlar ilk denemelerinde GAP'in çok kötü çalıştığını görmüş, ama sebebin GAP ya da fazladan token olmadığını, **öğrenme oranı** olduğunu bulmuştur. Uygun öğrenme oranıyla ikisi benzer sonuç verir (Ek D.3, Şekil 9). Sonraki çalışmalar (ör. MAE) çoğunlukla GAP kullanır. Aşama d'de iki seçeneği ve konumsuz ViT'i küçük bir veri kümesinde karşılaştırıyorsun.

**Yüksek çözünürlükte ince ayar.** Ön eğitim 224 × 224'te, ince ayar çoğu zaman daha yüksek çözünürlükte (ör. 384) yapılır. Yama boyu sabit kaldığı için dizi uzar: 196 yerine 576 yama. Transformer değişken uzunluklu diziyi işleyebilir (bellek sınırına kadar), ama önceden eğitilmiş konum embedding'leri 14 × 14'lük bir ızgara içindir. Yeni ızgaradaki her konumun embedding'i, **orijinal görüntüdeki yerine** göre eski embedding'lerin 2-B interpolasyonuyla elde edilir. ViT'te 2-B yapının elle eklendiği iki yerden biri budur; diğeri yamalara bölmektir.

:::türetim Bilineer interpolasyon: hangi koordinat?
Eski ızgara $g\times g$, yeni ızgara $g'\times g'$. Piksel merkezlerinin hizalanması (`align_corners=False`): yeni ızgaranın $i$. hücresinin merkezi, görüntü kesri olarak $(i+0.5)/g'$'dedir. Eski ızgarada bu kesir $u\,g-0.5$ indisine karşılık gelir:
$$i_{\text{eski}}=\frac{(i+0.5)\,g}{g'}-0.5 ,$$
kenarlarda $[0,g-1]$'e kırpılır. Embedding, bu koordinatı çevreleyen dört eski embedding'in bilineer ağırlıklı ortalamasıdır. $g=14$, $g'=24$ için yeni ilk hücre $0.5\cdot14/24-0.5=-0.21\to0$ (kenar), yeni 12. hücre $12.5\cdot14/24-0.5=6.79$, yani 6. ve 7. eski hücrelerin $0.21/0.79$ karışımı. Ek x1 bunu döngüsüz yazıyor. PyTorch: `F.interpolate(pos.reshape(1, g, g, D).permute(0, 3, 1, 2), size=(G, G), mode="bilinear", align_corners=False)`. [class] tokenının embedding'i interpolasyona katılmaz, olduğu gibi kalır.
:::

Aşağıdaki deneyde pürüzsüz bir "öğrenilmiş" konum alanını yeni ızgaraya bilineer, en yakın komşu ve yanlış yol olarak ham sırayla taşı. Görüntüdeki aynı yere karşılık gelen ideal embedding'e ne kadar yaklaştığını ölç.

::lab l2

### Endüktif yanlılık ve veri ölçeği {#yanlilik}

**Endüktif yanlılık** (inductive bias), bir modelin veriden bağımsız olarak "dünya şöyledir" diye varsaydığı yapıdır. CNN'de (M4.1) üç varsayım her katmana gömülüdür:
- **yerellik:** bir çıktı yalnız küçük bir komşuluğa bağlıdır;
- **2-B komşuluk yapısı:** pikseller bir ızgaradadır;
- **kaydırma eşdeğerliği:** aynı çekirdek her yerde kullanılır, nesne kayınca öznitelik de kayar.

ViT'te bu yanlılık çok daha azdır. Yalnız MLP katmanları yereldir (konum başına) ve kaydırma eşdeğerlidir; öz-dikkat **küreseldir**. 2-B komşuluk yapısı çok az kullanılır: yalnız başta yamalara bölerken ve ince ayarda konum embedding'lerini uyarlarken. Başlangıçtaki konum embedding'leri 2-B konum hakkında hiçbir bilgi taşımaz; yamalar arasındaki tüm uzamsal ilişkiler sıfırdan öğrenilmelidir.

:::sezgi Yanlılık neden az veride yardım eder, çok veride engel olur?
Doğru bir varsayım, modelin veriden öğrenmesi gereken şeyi azaltır: bir CNN "kenar her yerde kenardır" bilgisini mimarisinden alır, ViT bunu binlerce örnekten çıkarmak zorundadır. Az veride bu avantaj belirleyicidir. Ama varsayım her zaman tam doğru değildir. Uzak bölgeler arasındaki ilişkiler (ör. bir nesnenin iki ucu) CNN'de ancak çok katman sonra birleşir; ViT'te ilk katmanda bile birleşebilir. Veri yeterince büyüyünce esnek model, varsayımların dışındaki desenleri de öğrenir ve öne geçer. Yazarların ifadesiyle büyük ölçekli eğitim endüktif yanlılığı yener.
:::

Makalenin ana deneyi bu sezgiyi doğrular. Orta büyüklükte veride (ImageNet, 1.3M görüntü) güçlü düzenlileştirme olmadan eğitilen ViT'ler, benzer büyüklükteki ResNet'lerin birkaç puan gerisinde kalır. Ön eğitim verisi büyüdükçe (ImageNet-21k: 14M görüntü, 21k sınıf; JFT-300M: 303M görüntü, 18k sınıf) tablo tersine döner.

<figure class="fig wide"><svg viewBox="0 0 640 274" role="img" aria-label="Dört ViT modelinin ImageNet doğruluğu üç ön eğitim veri kümesine göre: küçük veride büyük modeller daha kötü, büyük veride daha iyi"><line x1="80" y1="230" x2="400" y2="230" class="ink"/><line x1="80" y1="30" x2="80" y2="230" class="ink"/><line x1="80" y1="230.0" x2="400" y2="230.0" class="mute"/><text x="74" y="234.0" text-anchor="end" class="sm">70</text><line x1="80" y1="180.0" x2="400" y2="180.0" class="mute"/><text x="74" y="184.0" text-anchor="end" class="sm">75</text><line x1="80" y1="130.0" x2="400" y2="130.0" class="mute"/><text x="74" y="134.0" text-anchor="end" class="sm">80</text><line x1="80" y1="80.0" x2="400" y2="80.0" class="mute"/><text x="74" y="84.0" text-anchor="end" class="sm">85</text><line x1="80" y1="30.0" x2="400" y2="30.0" class="mute"/><text x="74" y="34.0" text-anchor="end" class="sm">90</text><text x="80.0" y="248" text-anchor="middle" class="sm">ImageNet (1.3M)</text><text x="240.0" y="248" text-anchor="middle" class="sm">ImageNet-21k (14M)</text><text x="400.0" y="248" text-anchor="middle" class="sm">JFT-300M (303M)</text><text x="240.0" y="266" text-anchor="middle" class="sm">ön eğitim veri kümesi</text><text x="22" y="130.0" transform="rotate(-90 22 130.0)" text-anchor="middle" class="sm">ImageNet top-1 (%)</text><path d="M80.0,150.9 L240.0,90.3 L400.0,88.5" class="ca w2 " fill="none"/><circle cx="80.0" cy="150.9" r="3.5" class="ca-f"/><circle cx="240.0" cy="90.3" r="3.5" class="ca-f"/><circle cx="400.0" cy="88.5" r="3.5" class="ca-f"/><line x1="430" y1="50" x2="452" y2="50" class="ca w2 "/><text x="458" y="54" class="sm">ViT-B/16</text><path d="M80.0,164.7 L240.0,78.5 L400.0,58.8" class="cb w2 " fill="none"/><circle cx="80.0" cy="164.7" r="3.5" class="cb-f"/><circle cx="240.0" cy="78.5" r="3.5" class="cb-f"/><circle cx="400.0" cy="58.8" r="3.5" class="cb-f"/><line x1="430" y1="72" x2="452" y2="72" class="cb w2 "/><text x="458" y="76" class="sm">ViT-L/16</text><path d="M80.0,196.2 L240.0,117.2 L400.0,122.7" class="ca w2 dash" fill="none"/><circle cx="80.0" cy="196.2" r="3.5" class="ca-f"/><circle cx="240.0" cy="117.2" r="3.5" class="ca-f"/><circle cx="400.0" cy="122.7" r="3.5" class="ca-f"/><line x1="430" y1="94" x2="452" y2="94" class="ca w2 dash"/><text x="458" y="98" class="sm">ViT-B/32</text><path d="M80.0,218.4 L240.0,120.1 L400.0,86.3" class="cb w2 dash" fill="none"/><circle cx="80.0" cy="218.4" r="3.5" class="cb-f"/><circle cx="240.0" cy="120.1" r="3.5" class="cb-f"/><circle cx="400.0" cy="86.3" r="3.5" class="cb-f"/><line x1="430" y1="116" x2="452" y2="116" class="cb w2 dash"/><text x="458" y="120" class="sm">ViT-L/32</text><text x="430" y="160" class="sm">ImageNet’de L/16 &lt; B/16 (76.5 &lt; 77.9);</text><text x="430" y="176" class="sm">JFT’de L/16 &gt; B/16 (87.1 &gt; 84.2).</text><text x="430" y="198" class="sm">Büyük model + az veri = aşırı uyum.</text></svg><figcaption>Şekil 2 — Makalenin Tablo 5'indeki sayılar: modeller önce ImageNet, ImageNet-21k ya da JFT-300M'de ön eğitilip sonra ImageNet'te ince ayarlandı. Yalnız ImageNet'le büyük model (L) küçükten (B) kötüdür; veri büyüdükçe sıralama tersine döner. Yazarların ifadesiyle büyük ölçekli eğitim endüktif yanlılığı yener.</figcaption></figure>

:::örnek Sonuçlar (makalenin Tablo 2'si, ImageNet top-1, %)
| model | ön eğitim | ImageNet | ön eğitim maliyeti (TPUv3-çekirdek-gün) |
|---|---|---|---|
| ViT-H/14 | JFT-300M | **88.55** | 2.5k |
| ViT-L/16 | JFT-300M | 87.76 | 0.68k |
| ViT-L/16 | ImageNet-21k | 85.30 | 0.23k |
| BiT-L (ResNet152x4) | JFT-300M | 87.54 | 9.9k |
| Noisy Student (EfficientNet-L2) | JFT-300M | 88.4 / 88.5 | 12.3k |

ViT-H/14, en iyi CNN'leri geçti ya da onlara eşit oldu ve bunu **4–5 kat daha az** ön eğitim hesabıyla yaptı. Herkese açık ImageNet-21k ile ön eğitilmiş ViT-L/16 da (85.3) iyi sonuç verdi.
:::

**Hibrit mimari.** Yamalar ham pikseller yerine bir CNN'in öznitelik haritasından alınabilir. $1\times1$ "yamalar" kullanılırsa her öznitelik haritası konumu bir token olur. Makalede hibritler küçük modellerde saf ViT'ten iyidir; model büyüdükçe fark kapanır.

**İçeride ne öğreniliyor?** Makalenin Şekil 7'si üç şey gösterir:
1. Yama embedding'inin (E) ilk temel bileşenleri, her yamanın iç yapısı için düşük boyutlu temel fonksiyonlara benzer: CNN'lerin ilk katman filtrelerine benzer kenar ve renk desenleri.
2. Konum embedding'leri 2-B ızgarayı yansıtır (yukarıda).
3. **Ortalama dikkat uzaklığı**: her baş için, sorgu ve anahtar yamalar arasındaki piksel uzaklığının dikkat ağırlıklarıyla ortalaması. Bu ölçü CNN'deki alıcı alanın karşılığıdır. İlk katmanlarda bazı başlar zaten görüntünün çoğuna bakar, yani bilgiyi küresel olarak bütünleştirme yeteneği en alt katmanlarda bile kullanılır. Diğer başlar alt katmanlarda sürekli küçük uzaklıklarda kalır: model, CNN'in mimarisine gömülü yerelliği kendisi **öğrenir**. Uzaklık derinlikle artar. Ek c'deki `mean_attention_distance` bu ölçüyü hesaplıyor.

**Öz-gözetimli ön eğitim.** Makale, BERT'in maskeli dil modellemesine (M4.16) benzer bir **maskeli yama tahmini** de denedi. Bu yöntemle ViT-B/16 ImageNet'te %79.9'a ulaştı: sıfırdan eğitime göre %2 iyi, gözetimli ön eğitimin %4 gerisinde. Bu boşluğu sonradan MAE (He ve ark. 2022) kapattı. MAE yamaların %75'ini maskeler, kodlayıcı yalnız görünür yamaları işler, hafif bir kod çözücü pikselleri yeniden üretir. Konum için sabit 2-B sin-cos embedding'leri kullanılır (ek x1'deki `sincos_2d`).

### Az veriyle ViT: DeiT {#deit}

ViT'in "çok veri gerekir" sonucu, yalnız ImageNet'le iyi eğitilemeyeceği anlamına mı gelir? Touvron ve ark. (2021, DeiT) **eğitim tarifinin** de en az veri kadar önemli olduğunu gösterdi. Kullandıkları bileşenler: güçlü veri artırma (RandAugment, Mixup, CutMix, rastgele silme), tekrarlı artırma, stokastik derinlik, uzun eğitim ve AdamW. Bu tarifle mimarisi ViT-B ile aynı olan DeiT-B, **yalnız ImageNet-1k** ile %81.8 top-1'e ulaştı; ViT makalesinde aynı veriyle bildirilen değer %77.9'du. Eğitim tek bir 8 GPU'lu makinede üç günden kısa sürdü.

DeiT ayrıca dikkat yoluyla **damıtma** önerdi. [class] tokenının yanına bir **damıtma tokenı** eklenir. Bu token, öğretmen bir CNN'in (RegNetY) tahminini hedef alır; iki tokenın çıktıları çıkarımda birleştirilir. CNN öğretmen, ViT öğrenciye kendi endüktif yanlılığını tahminleri yoluyla "aktarır". Damıtmalı ve 384 çözünürlükte ince ayarlı DeiT-B %85.2'ye ulaştı. Sonuç: endüktif yanlılığın eksikliği kısmen veri artırma ve düzenlileştirmeyle, kısmen de bir öğretmenle telafi edilebilir.

### Hiyerarşi ve yerel pencereler: Swin {#swin}

ViT'in iki pratik sınırı vardır:
- **Karesel maliyet.** Yama sayısı çözünürlüğün karesiyle, dikkat maliyeti yama sayısının karesiyle büyür. Bölütleme ve tespit gibi yoğun tahmin görevlerinde (M4.9, M4.10) yüksek çözünürlük gerekir; bu maliyet orada çok ağırdır.
- **Tek ölçek.** Tüm katmanlarda 16 × 16 piksellik tek bir ölçek vardır; CNN'lerdeki gibi bir öznitelik piramidi (FPN, M4.9) yoktur.

**Swin Transformer** (Liu ve ark. 2021) iki çözüm getirdi:
1. **Pencere dikkati:** her token yalnız kendi $M\times M$'lik penceresindeki tokenlara bakar ($M=7$).
2. **Yama birleştirme:** her aşamanın sonunda $2\times2$ komşu token birleştirilip çözünürlük yarıya iner, kanal sayısı iki katına çıkar. Böylece CNN benzeri bir hiyerarşi oluşur.

:::türetim Küresel ve pencere dikkatinin maliyeti (Swin, Denk. 1–2)
$h\times w$ token, $C$ kanal. Küresel çok başlı öz-dikkat:
- $\mathbf Q,\mathbf K,\mathbf V$ ve çıktı izdüşümleri: $4hwC^2$;
- $\mathbf Q\mathbf K^\top$ ve ağırlıklı toplam: $2(hw)^2C$.

Toplam $\Omega_{\text{küresel}}=4hwC^2+2(hw)^2C$. Pencere dikkatinde her biri $M^2$ token içeren $hw/M^2$ pencere vardır; her pencerede skor maliyeti $2(M^2)^2C$, toplamı $\frac{hw}{M^2}\cdot2M^4C=2M^2hwC$:
$$\Omega_{\text{pencere}}=4hwC^2+2M^2hwC .$$
Karesel terim $(hw)^2$ yerine $M^2\cdot hw$ olur: token sayısında **doğrusal**. $56\times56$ token, $C=96$, $M=7$ için skor terimleri $2\cdot3136^2\cdot96\approx1.9\times10^9$'a karşı $2\cdot49\cdot3136\cdot96\approx2.9\times10^7$: 64 kat daha az.
:::

Pencereler birbirinden yalıtılmışsa bilgi pencere sınırını hiç geçemez. Swin bunu her ikinci katmanda pencereleri $\lfloor M/2\rfloor$ kaydırarak çözer: bir katmanın sınırları, bir sonrakinin pencerelerinin ortasına düşer. Uygulamada görüntü döngüsel olarak kaydırılır (`torch.roll`) ve kaydırma sonucu yan yana gelen ama orijinal görüntüde komşu olmayan bölgeler bir maskeyle ayrılır. Böylece pencere sayısı aynı kalır (ek x3).

Aşağıdaki deneyde 3 × 3 konvolüsyon, ViT, Swin ve kaydırmasız pencere için bir çıktı yamasının kaç katman sonra hangi girdilerden etkilenebildiğini gör ve katman başına dikkat maliyetini karşılaştır.

::lab l3

### Eğitim tarifi {#egitim}

ViT makalesinin tarifi (Ek B):
- **Ön eğitim:** Adam ($\beta_1=0.9$, $\beta_2=0.999$), batch 4096, yüksek ağırlık bozunumu (0.1; ImageNet-21k'de 0.03), doğrusal ısınma ve sonrasında doğrusal ya da kosinüs azalma. Yazarlar ResNet'ler için bile bu ayarda Adam'ın SGD'den biraz iyi olduğunu bildirir.
- **İnce ayar:** momentumlu SGD (0.9), batch 512, ağırlık bozunumu yok, gradyan normu 1'de kırpılır. Çözünürlük çoğunlukla 384 (ViT-H/14 için 518).
- **Değerlendirme:** önceden eğitilmiş baş atılır, sıfırla başlatılmış $D\times K$'lık bir doğrusal katman eklenir.

### PyTorch'ta {#pytorch}

| kavram | PyTorch | not |
|---|---|---|
| yama embedding'i | `nn.Conv2d(C, D, kernel_size=P, stride=P)` → `.flatten(2).transpose(1, 2)` | `(B, N, D)`; `F.unfold` + `@ W.T` ile aynı |
| [class] tokenı | `self.cls = nn.Parameter(torch.zeros(1, 1, D))`; `torch.cat([cls.expand(B, -1, -1), x], 1)` | |
| konum embedding'i | `self.pos = nn.Parameter(torch.zeros(1, N + 1, D))`; `trunc_normal_(std=0.02)` | öğrenilen 1-B |
| kodlayıcı | `nn.TransformerEncoderLayer(D, h, 4 * D, activation="gelu", norm_first=True, batch_first=True)` | Pre-LN |
| konum interpolasyonu | `F.interpolate(..., mode="bilinear", align_corners=False)` | [class] hariç |
| hazır modeller | `torchvision.models.vit_b_16(weights="IMAGENET1K_V1")`, `timm.create_model("vit_base_patch16_224")` | |
| pencere bölme | `x.view(B, H//M, M, W//M, M, C).permute(0, 1, 3, 2, 4, 5)` | ek x3 |
| kaydırma | `torch.roll(x, shifts=(-s, -s), dims=(1, 2))` | + maske |

Aşamalar bu sırayı izler. b: saf Python'da yama sayısı, parametre sayımı, dikkat FLOP'u ve yamalara bölme. c: NumPy'da `patchify`/`unpatchify`, konvolüsyon = doğrusal eşitliği, ortalama dikkat uzaklığı. d: PyTorch'ta küçük bir ViT ile CNN'in az veride karşılaştırması, [class] ile GAP ve konumsuz ViT. e: kopyasız yama embedding'i ve yama boyuna göre süre.

### Özet {#ozet}

- Görsel dikkat (Xu ve ark. 2015): CNN öznitelik haritasının 196 konumu "kaynak cümle" gibi kullanılır; kelime başına bir dikkat haritası. Yumuşak dikkat türevlenebilir, sert dikkat REINFORCE ister.
- ViT: $N=HW/P^2$ yama → doğrusal izdüşüm (= adımı $P$ olan konvolüsyon) + [class] + öğrenilen 1-B konum → Pre-LN Transformer kodlayıcı → [class] çıktısından sınıflandırma. 224/16: 196 yama, 197 token.
- ViT-B/16: katman 7 087 872, toplam 86 567 656 parametre; görüntüye özgü kısım %1'in altında.
- Konum: yokken yama torbası (0.614), 1-B / 2-B / göreli neredeyse eşit (≈ 0.64). [class] ve GAP doğru öğrenme oranıyla eşit.
- İnce ayarda çözünürlük artınca konum embedding'leri 2-B bilineer interpolasyonla taşınır, ham sırayla değil. 384'te dikkat maliyeti 8.6 kat.
- Endüktif yanlılık: CNN yerellik + kaydırma eşdeğerliği taşır, ViT taşımaz. Az veride CNN önde (ImageNet'te ViT-L/16 76.5 < ViT-B/16 77.9), çok veride ViT önde (JFT'de 87.1 > 84.2; ViT-H/14 88.55, BiT-L'nin 1/4 hesabıyla).
- DeiT: güçlü artırma + düzenlileştirme + damıtma tokenı ile yalnız ImageNet'te 81.8 (damıtmalı, 384: 85.2).
- Swin: $M\times M$ pencere dikkati, maliyet $4hwC^2+2M^2hwC$ (doğrusal); kaydırılmış pencerelerle bilgi pencere sınırını geçer; yama birleştirmeyle hiyerarşi.
