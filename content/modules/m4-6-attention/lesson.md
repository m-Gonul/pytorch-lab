### Problem: uzun yol {#problem}

M4.4'teki kodlayıcı–kod çözücü RNN uzun cümleleri çevirmekte çok kötüdür; sorun yine RNN'lerin sınırlı kısa süreli belleğidir. "I drink milk" → "Je bois du lait" çevirisinde "milk" kelimesinden çevirisi "lait"e giden yol **5 zaman adımı** uzunluğundadır: bu kelimenin temsili (diğer tüm kelimelerle birlikte) kullanılmadan önce birçok adım boyunca taşınmalıdır. Tüm kaynak cümle kodlayıcının **tek bir son durum vektörüne** sıkıştırılır — bir darboğaz. Bu yolu kısaltamaz mıyız?

Bu, Dzmitry Bahdanau ve ark.'nın çığır açan 2014 makalesinin ("Neural Machine Translation by Jointly Learning to Align and Translate") ana fikriydi: kod çözücünün her adımda (kodlayıcının kodladığı) **uygun kelimelere odaklanmasını** sağlayan bir teknik. Örneğin kod çözücü "lait"i üreteceği adımda dikkatini "milk"e yöneltir: bir girdi kelimesinden çevirisine giden yol artık **tek adımdır**, RNN'lerin kısa süreli bellek sınırları çok daha az etkilidir. Dikkat mekanizmaları sinirsel makine çevirisinde (ve genel olarak NLP'de) devrim yarattı; özellikle uzun cümlelerde (30 kelimenin üstü) son teknolojiyi belirgin biçimde ileri taşıdı.

**Mekanizma.** Kodlayıcının yalnız son gizli durumunu değil, **tüm çıktılarını** kod çözücüye gönder. Her adımda kod çözücü bu çıktıların ağırlıklı toplamını hesaplar: $\alpha_{(t,i)}$, $t$. kod çözücü adımında $i$. kodlayıcı çıktısının ağırlığıdır. $\alpha_{(3,2)}$ ağırlığı $\alpha_{(3,0)}$ ve $\alpha_{(3,1)}$'den çok büyükse kod çözücü 3. adımda 2. kelimeye ("milk") diğer ikisinden çok daha fazla dikkat eder. Ağırlıkları, modelin geri kalanıyla **birlikte eğitilen** küçük bir **hizalama modeli** (dikkat katmanı) üretir: her kodlayıcı çıktısı için bir **skor** (enerji) $e_{(t,i)}$ — o çıktının kod çözücünün durumuyla ne kadar "hizalı" olduğu — ve skorların softmax'ı. Belirli bir kod çözücü adımının tüm ağırlıklarının toplamı 1'dir.

| ad | skor $e_{(t,i)}$ | kaynak |
|---|---|---|
| toplamsal (concat, Bahdanau) | $\mathbf v^\top\tanh\big(\mathbf W[\mathbf h_{(t)};\mathbf y_{(i)}]\big)$ | Bahdanau ve ark. 2014 |
| nokta çarpımı (dot, Luong) | $\mathbf h_{(t)}^\top\mathbf y_{(i)}$ | Luong ve ark. 2015 |
| genel (general, Luong) | $\mathbf h_{(t)}^\top\mathbf W\mathbf y_{(i)}$ | Luong ve ark. 2015 |

$\mathbf h_{(t)}$: kod çözücünün durumu (Bahdanau bir önceki adımınkini, Luong o anki adımınkini kullanır), $\mathbf y_{(i)}$: $i$. kodlayıcı çıktısı; $\alpha_{(t,i)}=\mathrm{softmax}_i(e_{(t,i)})$, bağlam vektörü $\tilde{\mathbf h}_{(t)}=\sum_i\alpha_{(t,i)}\mathbf y_{(i)}$. Luong ve ark. nokta çarpımı türlerinin toplamsal dikkatten daha iyi çalıştığını gördü (nokta çarpımı iyi bir benzerlik ölçüsüdür ve modern donanımda çok daha hızlıdır; iki vektörün aynı boyutta olması gerekir); toplamsal dikkat bu yüzden bugün çok daha az kullanılır. Girdi $n$, çıktı da yaklaşık $n$ kelimeyse model $\approx n^2$ ağırlık hesaplar — cümleler binlerce kelime içermediği sürece yönetilebilir. Seq2seq ayrıntıları, beam search ve hizalama haritaları M4.13'te.

### Dikkat = türevlenebilir sözlük araması {#sozluk}

Kodlayıcı "They played chess" cümlesini analiz edip "They"in özne, "played"in fiil olduğunu anlamış ve bunu kelimelerin temsillerine kodlamış olsun. Kod çözücü özneyi çevirmiş, sırada fiil var: girdi cümlesinden fiili **getirmesi** gerekir. Bu bir sözlük aramasına benzer: kodlayıcı {"özne": "They", "fiil": "played", …} gibi bir sözlük oluşturmuş, kod çözücü "fiil" anahtarına karşılık gelen değeri istiyor.

<figure class="fig wide"><svg viewBox="0 0 620 240" role="img" aria-label="Sorgu vektörü her anahtarla karşılaştırılır, softmax ağırlıkları değerlerin ağırlıklı ortalamasını belirler"><defs><marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="20" y="20" width="150" height="30" rx="5" class="soft-b cb"/><text x="95" y="39" text-anchor="middle" class="sm">sorgu q: "fiil arıyorum"</text><rect x="20" y="80" width="110" height="30" rx="5" class="soft-a ca"/><text x="75" y="99" text-anchor="middle" class="sm">anahtar k: özne</text><rect x="300" y="80" width="120" height="30" rx="5" class="soft-p cp"/><text x="360" y="99" text-anchor="middle" class="sm">değer v: They</text><text x="210" y="99" text-anchor="middle" class="sm">q·k → α = 0.03</text><rect x="160" y="104" width="3" height="6" class="cq-f"/><line x1="130" y1="95" x2="158" y2="95" class="ink" marker-end="url(#a1)"/><line x1="262" y1="95" x2="298" y2="95" class="ink" marker-end="url(#a1)"/><rect x="20" y="124" width="110" height="30" rx="5" class="soft-a ca"/><text x="75" y="143" text-anchor="middle" class="sm">anahtar k: fiil</text><rect x="300" y="124" width="120" height="30" rx="5" class="soft-p cp"/><text x="360" y="143" text-anchor="middle" class="sm">değer v: played</text><text x="210" y="143" text-anchor="middle" class="sm">q·k → α = 0.94</text><rect x="160" y="148" width="94" height="6" class="cq-f"/><line x1="130" y1="139" x2="158" y2="139" class="ink" marker-end="url(#a1)"/><line x1="262" y1="139" x2="298" y2="139" class="ink" marker-end="url(#a1)"/><rect x="20" y="168" width="110" height="30" rx="5" class="soft-a ca"/><text x="75" y="187" text-anchor="middle" class="sm">anahtar k: nesne</text><rect x="300" y="168" width="120" height="30" rx="5" class="soft-p cp"/><text x="360" y="187" text-anchor="middle" class="sm">değer v: chess</text><text x="210" y="187" text-anchor="middle" class="sm">q·k → α = 0.03</text><rect x="160" y="192" width="3" height="6" class="cq-f"/><line x1="130" y1="183" x2="158" y2="183" class="ink" marker-end="url(#a1)"/><line x1="262" y1="183" x2="298" y2="183" class="ink" marker-end="url(#a1)"/><line x1="95" y1="50" x2="95" y2="78" class="ink" marker-end="url(#a1)"/><rect x="470" y="124" width="130" height="30" rx="5" class="soft-p cp"/><text x="535" y="143" text-anchor="middle" class="sm">Σ α·v ≈ v(played)</text><line x1="420" y1="95" x2="468" y2="139" class="ink" marker-end="url(#a1)"/><line x1="420" y1="139" x2="468" y2="139" class="ink" marker-end="url(#a1)"/><line x1="420" y1="183" x2="468" y2="139" class="ink" marker-end="url(#a1)"/><text x="20" y="230" class="sm">Gerçek sözlükte anahtar ya eşleşir ya eşleşmez; burada her anahtar benzerliği oranında katkı verir → türevlenebilir.</text></svg><figcaption>Şekil 1 — Dikkat, türevlenebilir bir sözlük araması gibi. Kodlayıcı cümleyi {"özne": They, "fiil": played, …} gibi bir sözlüğe çevirmiş olsun; kod çözücü fiili çevirmek üzereyken "fiil" anahtarını arar. Ama anahtarlar ve sorgu öğrenilmiş vektörlerdir, tam eşleşme yoktur: sorgu her anahtarla nokta çarpımıyla karşılaştırılır, softmax benzerlikleri toplamı 1 olan ağırlıklara çevirir, çıktı değerlerin ağırlıklı toplamıdır (Géron'un anlatımından).</figcaption></figure>

Ama modelin anahtarları temsil eden "özne" ya da "fiil" gibi ayrık tokenları yoktur; bu kavramların eğitimde öğrenilmiş **vektör** temsilleri vardır. Dolayısıyla aramada kullanacağı anahtar (**sorgu**, query) sözlükteki hiçbir anahtarla **tam eşleşmez**. Çözüm: sorgu ile sözlükteki her anahtar arasında bir benzerlik ölçüsü hesapla, softmax ile bu skorları toplamı 1 olan ağırlıklara çevir. "Fiil" anahtarı sorguya açık ara en benzerse ağırlığı 1'e yakın olur; model değerlerin ağırlıklı toplamını alır ve sonuç "played"in temsiline çok yakın olur. Kısacası bütün süreci **türevlenebilir bir sözlük araması** olarak düşünebilirsin — türevlenebilir olduğu için gradyan inişiyle öğrenilebilir.

### Ölçekli nokta çarpımı dikkati {#sdpa}

Transformer'ın kullandığı benzerlik ölçüsü Luong dikkatindeki gibi yalnız nokta çarpımıdır; denklem de bir ölçek çarpanı dışında aynıdır:

:::tanım Ölçekli nokta çarpımı dikkati (Vaswani ve ark. 2017)
$$\mathrm{Attention}(\mathbf Q,\mathbf K,\mathbf V)=\mathrm{softmax}\Big(\frac{\mathbf Q\mathbf K^\top}{\sqrt{d_{\text{keys}}}}\Big)\mathbf V$$
- $\mathbf Q$: sorgu başına bir satır, shape $[n_{\text{queries}},d_{\text{keys}}]$ ($d_{\text{keys}}$: her sorgunun ve anahtarın boyutu).
- $\mathbf K$: anahtar başına bir satır, $[n_{\text{keys}},d_{\text{keys}}]$; $\mathbf V$: değer başına bir satır, $[n_{\text{keys}},d_{\text{values}}]$ (anahtar ve değer sayısı aynı).
- $\mathbf Q\mathbf K^\top$: $[n_{\text{queries}},n_{\text{keys}}]$ — her sorgu/anahtar çifti için bir benzerlik skoru. Softmax (satır satır) aynı shape'i korur, her satırın toplamı 1 olur.
- Çıktı: $[n_{\text{queries}},d_{\text{values}}]$ — sorgu başına bir satır, her satır sorgunun sonucu (değerlerin ağırlıklı toplamı).
:::

:::türetim Neden $\sqrt{d_k}$ ile bölünür?
$\mathbf q,\mathbf k\in\mathbb R^{d_k}$'nın bileşenleri bağımsız, ortalaması 0, varyansı 1 olsun. $\mathbf q\cdot\mathbf k=\sum_{t=1}^{d_k}q_tk_t$; her terimin ortalaması $\mathbb E[q_t]\mathbb E[k_t]=0$, varyansı $\mathbb E[q_t^2]\mathbb E[k_t^2]=1$. Bağımsız terimlerin varyansları toplanır:
$$\mathrm{Var}(\mathbf q\cdot\mathbf k)=d_k,\qquad\mathrm{std}=\sqrt{d_k}.$$
$d_k=64$'te skorlar tipik olarak ±8 civarında dağılır; softmax'a bu büyüklükte farklar girince çıktı neredeyse one-hot olur (**doyar**) ve softmax'ın Jacobian'ı $\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top$ (M1.5) neredeyse sıfıra iner: gradyanlar küçülür, öğrenme durur. $\sqrt{d_k}$'ya bölmek skorların varyansını 1'e çeker. Bu bir **sıcaklık** seçimidir: $\mathrm{softmax}(\mathbf s/\tau)$ ile $\tau=\sqrt{d_k}$. (Varsayım: bileşenler birim varyanslı ve ilişkisiz; eğitimde bu bozulabilir — bu yüzden modern modeller ayrıca sorgu/anahtar normalizasyonu kullanabilir.) Ekteki deney (l1) bunu ölçüyor.
:::

:::örnek Ders slaytındaki öz-dikkat hesabı
"I enjoy walking everyday" cümlesi, $d_k=64$ ($\sqrt{d_k}=8$). "I" kelimesinin sorgusu $\mathbf q_1$'in dört anahtarla skorları: $80, 64, 48, 48$; 8'e bölünce $10, 8, 6, 6$; softmax: $e^{10}=22\,026$, $e^8=2\,981$, $e^6=403$ (iki kez) → toplam $25\,814$ → **0.853, 0.115, 0.016, 0.016** (slaytta 0.85, 0.12, 0.02, 0.02). Çıktı $\mathbf z_1=0.853\,\mathbf v_1+0.115\,\mathbf v_2+0.016\,\mathbf v_3+0.016\,\mathbf v_4$: "I"ın yeni temsili büyük ölçüde kendisi, biraz da "enjoy". "enjoy" için skorlar $40,56,40,32$ → $5,7,5,4$ → $0.10, 0.76, 0.10, 0.04$. Bu dört satır, 4×4'lük bir dikkat matrisinin satırlarıdır.
:::

**Öz-dikkat** (self-attention): kodlayıcıda denklem her girdi cümlesine $\mathbf Q$, $\mathbf K$ ve $\mathbf V$'nin **üçü de** o cümlenin kelimeleri olacak biçimde uygulanır — her kelime aynı cümledeki her kelimeyle (kendisi dahil) karşılaştırılır. Örneğin "They welcomed the Queen of the United Kingdom" cümlesinde "Queen" kelimesinin çıktısı tüm kelimelere bağlıdır, ama büyük olasılıkla "United" ve "Kingdom"a "They" ya da "welcomed"dan daha çok dikkat eder. Matris biçiminde: her kelimenin embedding'i $\mathbf x_i$ öğrenilen matrislerle sorguya, anahtara, değere izdüşürülür; satırları üst üste koyunca
$$\mathbf Q=\mathbf X\mathbf W_Q,\qquad\mathbf K=\mathbf X\mathbf W_K,\qquad\mathbf V=\mathbf X\mathbf W_V,\qquad \mathbf Z=\mathrm{softmax}\big(\mathbf Q\mathbf K^\top/\sqrt{d_k}\big)\mathbf V .$$
Soran (dikkat eden) kelime **sorgu** vektörünü, sorulan (dikkat edilen) kelimeler **anahtar** vektörlerini kullanır.

**Maskeler.** Bazı anahtar/değer çiftleri, softmax'tan hemen önce karşılık gelen skorlara çok büyük negatif bir değer ($-\infty$) eklenerek **maskelenebilir**: softmax'tan sonra ağırlıkları tam 0 olur.
- **Dolgu maskesi:** dolgu tokenlarına (M4.4) kimse dikkat etmemeli; aksi hâlde kod çözücü anlamsız kodlayıcı durumlarına bakabilir.
- **Nedensel maske:** kod çözücünün maskeli öz-dikkat katmanında her kelime yalnız **kendinden önceki** kelimelere bakabilir ($j>i$ skorları $-\infty$). Çıkarımda kod çözücü yalnız şimdiye kadar ürettiği kelimelere erişebilir; eğitimde de gelecekteki çıktı tokenlarını maskelemek gerekir. (Nedensel maskenin tüm diziyi tek ileri geçişte eğitmeyi mümkün kılması M4.7 ve M4.14'te.)
- **Çapraz dikkat** (kod çözücünün üst dikkat katmanı): anahtarlar ve değerler kodlayıcının çıktılarından, sorgular kod çözücünün tarafından gelir.

Aşağıdaki deneyde küçük bir cümlede dikkat matrisini gör: bir satıra (sorguya) tıkla, skorları ve çıktının hangi değerlerden karıştığını incele; nedensel ve dolgu maskelerini, ölçeklemeyi ve sıcaklığı değiştir.

::lab l2

Ardından $\sqrt{d_k}$ ile bölmenin softmax doygunluğunu ve gradyanı nasıl etkilediğini ölç:

::lab l1

:::türetim Dikkatin geri geçişi
$\mathbf S=\mathbf Q\mathbf K^\top/\sqrt{d_k}$, $\mathbf P=\mathrm{softmax}_{\text{satır}}(\mathbf S)$, $\mathbf O=\mathbf P\mathbf V$, $\mathbf G=\partial L/\partial\mathbf O$. Matris çarpımı kuralıyla (M0.2): $\partial L/\partial\mathbf V=\mathbf P^\top\mathbf G$, $\partial L/\partial\mathbf P=\mathbf G\mathbf V^\top$. Satır softmax'ının Jacobian'ı (M1.5) her satırda $\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top$ olduğundan:
$$\frac{\partial L}{\partial\mathbf S}=\mathbf P\odot\Big(\frac{\partial L}{\partial\mathbf P}-\mathrm{rowsum}\big(\tfrac{\partial L}{\partial\mathbf P}\odot\mathbf P\big)\Big),$$
(rowsum sütun vektörü, satır boyunca yayınlanır). Sonra $\partial L/\partial\mathbf Q=\frac{1}{\sqrt{d_k}}\frac{\partial L}{\partial\mathbf S}\mathbf K$, $\partial L/\partial\mathbf K=\frac1{\sqrt{d_k}}\big(\frac{\partial L}{\partial\mathbf S}\big)^\top\mathbf Q$. Maskeli konumlarda $P=0$ olduğundan gradyanları da 0'dır. Not: $\mathrm{rowsum}(\frac{\partial L}{\partial\mathbf P}\odot\mathbf P)=\mathrm{rowsum}(\mathbf G\odot\mathbf O)$ — FlashAttention geri geçişte $\mathbf P$'yi saklamak yerine bu eşitliği kullanır.
:::

### Çok başlı dikkat {#mha}

Tek bir ölçekli nokta çarpımı dikkat katmanı kullansaydık, bir kelime temsilinin tüm özelliklerini **tek seferde** sorgulayabilirdik. "They played chess" cümlesindeki "played": kodlayıcı onun fiil olduğunu kodlamış, ama temsil konum bilgisini (konum kodlamaları sayesinde) ve çeviri için yararlı başka birçok özelliği de (ör. geçmiş zaman olduğunu) içerir. **Çok başlı dikkat** (multi-head attention) katmanı, değerlerin, anahtarların ve sorguların **birden çok farklı doğrusal dönüşümünü** uygular: model kelime temsilini farklı alt uzaylara izdüşürür; belki bir doğrusal katman temsili yalnız "fiil olma" bilgisinin kaldığı bir alt uzaya, diğeri yalnız "geçmiş zaman" bilgisine izdüşürür. Ölçekli nokta çarpımı katmanları arama aşamasını yapar; sonra tüm sonuçlar birleştirilip orijinal uzaya geri izdüşürülür.

<figure class="fig wide"><svg viewBox="0 0 620 240" role="img" aria-label="Çok başlı dikkat: her baş girdinin farklı doğrusal izdüşümleri üzerinde ölçekli nokta çarpımı dikkati yapar; çıktılar birleştirilip bir doğrusal katmandan geçer"><defs><marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="20" y="200" width="90" height="28" rx="5" class="mute"/><text x="65" y="218" text-anchor="middle" class="sm">X (T × d)</text><rect x="150" y="150" width="120" height="26" rx="5" class="soft-a ca"/><text x="210" y="167" text-anchor="middle" class="sm">Q·W<tspan baseline-shift="sub" font-size="9">Q1</tspan>  K·W<tspan baseline-shift="sub" font-size="9">K1</tspan>  V·W<tspan baseline-shift="sub" font-size="9">V1</tspan></text><rect x="150" y="95" width="120" height="30" rx="5" class="soft-p cp"/><text x="210" y="114" text-anchor="middle" class="sm">ölçekli nokta çarpımı</text><line x1="210" y1="150" x2="210" y2="127" class="ink" marker-end="url(#a1)"/><line x1="110" y1="214" x2="210" y2="214" class="ink"/><line x1="210" y1="214" x2="210" y2="178" class="ink" marker-end="url(#a1)"/><text x="210" y="88" text-anchor="middle" class="sm">baş 1: T × d/h</text><line x1="210" y1="95" x2="375" y2="62" class="ink" marker-end="url(#a1)"/><rect x="300" y="150" width="120" height="26" rx="5" class="soft-a ca"/><text x="360" y="167" text-anchor="middle" class="sm">Q·W<tspan baseline-shift="sub" font-size="9">Q2</tspan>  K·W<tspan baseline-shift="sub" font-size="9">K2</tspan>  V·W<tspan baseline-shift="sub" font-size="9">V2</tspan></text><rect x="300" y="95" width="120" height="30" rx="5" class="soft-p cp"/><text x="360" y="114" text-anchor="middle" class="sm">ölçekli nokta çarpımı</text><line x1="360" y1="150" x2="360" y2="127" class="ink" marker-end="url(#a1)"/><line x1="110" y1="214" x2="360" y2="214" class="ink"/><line x1="360" y1="214" x2="360" y2="178" class="ink" marker-end="url(#a1)"/><text x="360" y="88" text-anchor="middle" class="sm">baş 2: T × d/h</text><line x1="360" y1="95" x2="375" y2="62" class="ink" marker-end="url(#a1)"/><rect x="450" y="150" width="120" height="26" rx="5" class="soft-a ca"/><text x="510" y="167" text-anchor="middle" class="sm">Q·W<tspan baseline-shift="sub" font-size="9">Q3</tspan>  K·W<tspan baseline-shift="sub" font-size="9">K3</tspan>  V·W<tspan baseline-shift="sub" font-size="9">V3</tspan></text><rect x="450" y="95" width="120" height="30" rx="5" class="soft-p cp"/><text x="510" y="114" text-anchor="middle" class="sm">ölçekli nokta çarpımı</text><line x1="510" y1="150" x2="510" y2="127" class="ink" marker-end="url(#a1)"/><line x1="110" y1="214" x2="510" y2="214" class="ink"/><line x1="510" y1="214" x2="510" y2="178" class="ink" marker-end="url(#a1)"/><text x="510" y="88" text-anchor="middle" class="sm">baş 3: T × d/h</text><line x1="510" y1="95" x2="375" y2="62" class="ink" marker-end="url(#a1)"/><rect x="300" y="30" width="150" height="30" rx="5" class="soft-b cb"/><text x="375" y="49" text-anchor="middle" class="sm">birleştir → · W<tspan baseline-shift="sub" font-size="9">O</tspan></text><text x="460" y="50" class="sm">çıktı (T × d)</text></svg><figcaption>Şekil 2 — Çok başlı dikkat (h = 3 baş gösteriliyor; makalede h = 8). Her baş kendi öğrenilen izdüşümleriyle girdinin farklı bir alt uzayına bakar ("fiil mi?", "zaman eki?", "hangi kişiye gönderme?"); çıktılar birleştirilip W<sub>O</sub> ile orijinal boyuta geri izdüşürülür. Baş başına boyut d/h olduğundan toplam hesap tek bir tam boyutlu başla aynıdır.</figcaption></figure>

:::tanım Çok başlı dikkat
$h$ baş, model boyu $d$, baş başına $d_k=d_v=d/h$:
$$\mathrm{head}_i=\mathrm{Attention}(\mathbf X\mathbf W_Q^{(i)},\ \mathbf X\mathbf W_K^{(i)},\ \mathbf X\mathbf W_V^{(i)}),\qquad\mathrm{MHA}(\mathbf X)=\big[\mathrm{head}_1;\dots;\mathrm{head}_h\big]\mathbf W_O .$$
$\mathbf W_Q^{(i)},\mathbf W_K^{(i)},\mathbf W_V^{(i)}\in\mathbb R^{d\times d/h}$, $\mathbf W_O\in\mathbb R^{d\times d}$. Toplam parametre (bias'sız) $4d^2$ — baş sayısından bağımsız. Uygulamada tüm başların izdüşümleri tek bir $d\times d$ matris çarpımıyla yapılır, sonuç $(N,T,h,d/h)$'ye yeniden şekillendirilip $(N,h,T,d/h)$'ye devrik alınır; her baş bağımsız bir dikkattir. Makalede $d=512$, $h=8$, $d_k=64$.
:::

Her dikkat başı girdi embedding'lerini farklı biçimde dönüştürdüğü için her birinin kendi $\mathbf Z$ çıktı matrisi vardır; slayttaki sezgisel örnek: bir baş zamirleri karşılık geldikleri kişi/nesneyle, bir başı sıfatları nitelediği isimle, bir başı fiili özneyle ilişkilendirebilir. (Eğitilmiş modellerde başların bir kısmı gerçekten böyle yorumlanabilir örüntüler gösterir, ama bu garanti değildir; birçok baş budanabilir — Michel ve ark. 2019.)

**Konum bilgisi.** Dikkat **permütasyona eşdeğişirdir**: girdi tokenlarının sırasını karıştırırsan çıktılar da aynı biçimde karışır — dikkat sıra bilmez. Kelimelerin konumunu modele ayrıca vermek gerekir: **konum kodlamaları** (sinüzoidal ya da öğrenilen, RoPE) — M4.14'te.

**Maliyet.** $T$ uzunluklu dizide skor matrisi $T\times T$: hesap $O(T^2d)$, bellek $O(T^2)$. Kısa cümlelerde sorun değil, ama 100 000 tokenlık bağlamda $10^{10}$ skor demek. Hesap kaçınılmazdır (her token her tokena bakar), ama **bellek** değil:

### Çevrimiçi softmax ve FlashAttention {#flash}

:::türetim Softmax'ı bloklar hâlinde hesaplamak
Tek bir sorgunun skorları $s_1,\dots,s_T$ ve değerleri $v_1,\dots,v_T$. Kararlı softmax önce maksimumu çıkarır (M1.5): $o=\sum_je^{s_j-m}v_j\big/\sum_je^{s_j-m}$, $m=\max_js_j$. Skorları bloklar hâlinde görüyoruz ve tüm $m$'yi baştan bilmiyoruz. Koşan değerleri tut: $m$ (şimdiye kadarki maksimum), $\ell=\sum e^{s_j-m}$, $\mathbf a=\sum e^{s_j-m}v_j$. Yeni bir blok gelince $m'=\max(m,\max_{\text{blok}}s_j)$ ve
$$\ell\leftarrow e^{m-m'}\ell+\sum_{j\in\text{blok}}e^{s_j-m'},\qquad\mathbf a\leftarrow e^{m-m'}\mathbf a+\sum_{j\in\text{blok}}e^{s_j-m'}v_j,\qquad m\leftarrow m'.$$
Eski toplamlar eski maksimuma göreydi; $e^{m-m'}$ çarpanı onları yeni maksimuma göre **yeniden ölçekler** ($e^{s-m}=e^{s-m'}e^{m'-m}$). Sonda $o=\mathbf a/\ell$ — tam softmax ağırlıklı toplamın **aynısı** (yaklaşık değil). Bellek: skor matrisinin tamamı yerine blok başına $O(B)$.
:::

**FlashAttention** (Dao ve ark. 2022) bu fikirle $\mathbf Q$, $\mathbf K$, $\mathbf V$'yi GPU'nun küçük ama çok hızlı paylaşılan belleğine (SRAM) sığan bloklar hâlinde işler; $T\times T$ matris yavaş ana belleğe (HBM) **hiç yazılmaz**. Aritmetik işlem sayısı aynıdır (hatta geri geçişte $\mathbf P$'yi yeniden hesapladığı için biraz fazladır), ama dikkat bellek trafiği sınırlı bir işlem olduğundan gerçek süre birkaç kat kısalır ve bellek $O(T)$'ye iner. PyTorch'un `F.scaled_dot_product_attention`'ı uygun donanımda bu çekirdeği otomatik seçer.

::lab l3

### PyTorch karşılıkları {#pytorch}

| kavram | PyTorch |
|---|---|
| ölçekli nokta çarpımı | `F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)`; shape `(N, h, T, d_k)`; FlashAttention/bellek-verimli çekirdekleri otomatik seçer |
| çok başlı dikkat | `nn.MultiheadAttention(embed_dim=d, num_heads=h, batch_first=True)`; `out, w = mha(q, k, v, key_padding_mask=pad, attn_mask=causal)` |
| maske anlamı | bool maskede `True` = **maskele** (`nn.MultiheadAttention`), ama `F.scaled_dot_product_attention`'da bool `True` = **katıl**; ondalıklı maske skora eklenir ($-\infty$ ya da 0) |
| nedensel maske | `torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)` (True = gelecek) ya da `is_causal=True` |
| ağırlıklar | `in_proj_weight` $(3d,d)$ = $[\mathbf W_Q;\mathbf W_K;\mathbf W_V]^\top$, `out_proj` = $\mathbf W_O$ |

Maskelerin iki API'de **zıt** anlamlı bool kurala sahip olması klasik bir hata kaynağıdır; şüphede ondalıklı ($-\infty$/0) maske kullan.

### Özet {#ozet}

- Dikkat, kodlayıcı–kod çözücüdeki tek vektörlük darboğazı kaldırır: kod çözücü her adımda tüm kodlayıcı çıktılarının ağırlıklı toplamını alır; ağırlıklar skorların softmax'ı (toplamsal / nokta / genel).
- Türevlenebilir sözlük araması: sorgu her anahtarla karşılaştırılır, softmax ağırlıkları değerleri karıştırır.
- $\mathrm{softmax}(\mathbf Q\mathbf K^\top/\sqrt{d_k})\mathbf V$; $\sqrt{d_k}$ skor varyansını 1'e çeker, softmax doygunluğunu önler.
- Öz-dikkat (Q = K = V kaynağı aynı), çapraz dikkat (K, V kodlayıcıdan), maskeler (dolgu, nedensel) $-\infty$ ile.
- Çok başlı: $h$ farklı izdüşüm, birleştir, $\mathbf W_O$; $4d^2$ parametre; dikkat sıra bilmez → konum kodlaması.
- $O(T^2)$ hesap kaçınılmaz, $O(T^2)$ bellek değil: çevrimiçi softmax ile bloklar hâlinde tam (yaklaşık olmayan) sonuç — FlashAttention.
