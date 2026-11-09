### Büyük dil modellerinin kısa tarihi {#tarih}

Doğal dil işleme (NLP), bilgisayarlara metni ve konuşmayı insanların anladığı biçimde anlama yeteneği kazandırmaya çalışan yapay zekâ dalıdır ve bilgisayar biliminin en eski alanlarından biridir: kökleri 1950'lere uzanır; ünlü Turing testi de "akıllı" ajanların sohbetine dayanır. Görevleri geniştir: konuşma tanıma ve sentezi, kelime bölütleme, sözdizimsel çözümleme, anlam görevleri, makine çevirisi, diyalog sistemleri, özetleme, duygu analizi, metin üretimi…

| dönem | yaklaşım / model | not |
|---|---|---|
| 1950'ler | sembolik, kural tabanlı | sınırlı başarı |
| 1990'lar | istatistiksel modelleme, makine öğrenmesi | ilk ticari başarılar |
| 2010'lar | sinirsel NLP: derin RNN/LSTM/GRU (M4.3–M4.4), seq2seq + dikkat (M4.6, M4.13) | |
| 2017 | **Transformer** (Vaswani ve ark.) | yalnız dikkat; RNN yok |
| 2018 | **GPT-1** (OpenAI, 117M), BERT (Google) | ön-eğitim + ince ayar |
| 2019 | **GPT-2** (1.5B) | sıfır örnekli görevler; bu modülün modeli |
| 2020 | **GPT-3** (175B) | bağlam içi öğrenme (few-shot) |
| 2022 | InstructGPT, **ChatGPT** (GPT-3.5 serisi üzerine RLHF) | sohbet ürünü |
| 2023 | GPT-4, Claude (Anthropic), LLaMA, Mistral 7B, Qwen | açık ağırlıklı modellerin yükselişi |
| 2024 | akıl yürütme modelleri: OpenAI o1, Gemini 2.0 Flash Thinking; DeepSeek-V3 (Aralık) | test anında daha uzun "düşünme" |
| 2025 | DeepSeek-R1, Gemma 3 (açık) | |
| 2026 | Claude Opus 4.8, Claude Mythos / Fable, GPT-5.5, DeepSeek V4 Pro (açık) | |

:::uyarı Kaynak düzeltmesi: zaman çizelgesi
Ders slaytındaki çizelgede birkaç hata var:
- "ChatGPT-1 (2018) / ChatGPT-2 (2019) / ChatGPT-3 (2020)" yerine doğrusu **GPT-1 / GPT-2 / GPT-3**: bunlar modellerin adlarıdır. **ChatGPT** Kasım 2022'de çıkan sohbet ürünüdür; GPT-3.5 serisinden bir modelin insan geri bildirimiyle pekiştirmeli öğrenme (RLHF) ile ince ayarlanmış hâlidir. "ChatGPT-4" de yanlıştır: model GPT-4'tür.
- "GPT-3.5: 175 milyar parametre": OpenAI GPT-3.5'in parametre sayısını açıklamadı; 175B GPT-3'ün sayısıdır.
- Mistral (Mistral 7B, Eylül 2023) ve Qwen (ilk açık ağırlıklı sürüm Ağustos 2023) 2024'te değil 2023'te çıktı. İlk yaygın "akıl yürütme modeli" OpenAI o1'dir (Eylül 2024); Gemini 2.0 Flash Thinking Aralık 2024'tedir.
- Gemma 3, 2026'nın değil **Mart 2025**'in modelidir.
- Yazım: "Antropic" → Anthropic, "Opes" → Opus.

2026 satırındaki model adları ve tarihleri hızla eskir; doğrulamak için şirketlerin kendi duyurularına bak.
:::

Bu modülde 2018–2019'un GPT'sini sıfırdan kuruyorsun. Sonrakilerin neredeyse hepsi aynı iskeletin büyütülmüş ve küçük değişikliklerle iyileştirilmiş hâlidir (RMSNorm, SwiGLU, RoPE, GQA — aşağıda notlar var).

### Kodlayıcı–kod çözücüden yalnız kod çözücüye {#mimari}

Dikkat mekanizmaları o kadar güçlüdür ki son teknoloji modelleri **yalnız dikkatle** kurabilirsin. Google araştırmacılarının 2017 tarihli "Attention Is All You Need" makalesi tam bunu yaptı: **Transformer** adlı mimari, hiç rekürent ya da konvolüsyonel katman kullanmadan, yalnız dikkat mekanizmaları artı gömme katmanları, yoğun (dense) katmanlar, normalizasyon katmanları ve birkaç başka parçayla makine çevirisinde son teknolojiyi belirgin biçimde ileri taşıdı. Üstelik eğitimi çok daha hızlı ve paralelleştirmesi çok daha kolaydı; önceki modellerin zamanının ve maliyetinin küçük bir kesrine eğitildi.

**Orijinal Transformer** (ayrıntıları M4.14'te) bir kodlayıcı–kod çözücüdür:

- **Kodlayıcı** (solda) kelime kimliklerinden oluşan bir cümle batch'i alır: girdi shape'i `[batch, en uzun girdi cümlesi]`; her kelimeyi 512 boyutlu bir temsile kodlar: çıktı `[batch, en uzun girdi cümlesi, 512]`. Üst kısım $N$ kez yığılır (makalede $N=6$).
- **Kod çözücü** (sağda) eğitimde hedef cümleyi alır, **bir adım sağa kaydırılmış** (başa bir dizi-başı token'ı eklenmiş) hâlde. Kodlayıcının son çıktısı (N kodlayıcı bloğundan sonra) **N kod çözücü bloğunun hepsine** verilir. Her zaman adımında sözlükteki her kelime için bir olasılık üretir: çıktı `[batch, en uzun çıktı cümlesi, sözlük boyu]`.
- Çıkarımda kod çözücüye hedef verilemez; önceki adımlarda ürettiği kelimeler verilir (dizi-başı token'ından başlayarak). **Bu bir RNN değildir**: model her turda yeniden çağrılır, her seferinde bir kelime daha tahmin eder, o kelime bir sonraki turda girdiye eklenir; dizi-sonu token'ı üretilene kadar sürer. (Bu tekrar çağrının maliyeti ve KV önbelleği aşağıda.)

Tanıdık parçalar: **2 gömme katmanı**; her biri ardından bir normalizasyon katmanı gelen **5 × N artık bağlantı** (kodlayıcıda blok başına 2, kod çözücüde 3); her biri iki yoğun katmandan oluşan **2 × N "ileri beslemeli" modül** (ilki ReLU, ikincisi aktivasyonsuz) ve en sonda softmax'lı yoğun çıkış katmanı. Bu katmanların hepsi **zamana dağıtılmıştır** (time-distributed): her kelime diğerlerinden bağımsız işlenir. O hâlde bir cümle, kelimelere tek tek bakılarak nasıl çevrilir? Yeni parçalar burada devreye girer: kelimeler arası ilişkiyi kuran **çok başlı dikkat** (M4.6) ve sıra bilgisini veren **konum gömmeleri**.

Transformer'ın üç ailesi bu iki yarıdan hangisini tuttuklarıyla ayrılır:

| aile | yapı | dikkat maskesi | ön-eğitim görevi | örnek |
|---|---|---|---|---|
| yalnız kodlayıcı | kodlayıcı yığını | çift yönlü (maske yok) | maskeli dil modeli | BERT (M4.16) |
| **yalnız kod çözücü** | kod çözücü yığını, **çapraz dikkat yok** | **nedensel** | bir sonraki token | **GPT** ailesi, LLaMA, Claude… |
| kodlayıcı–kod çözücü | ikisi + çapraz dikkat | kodlayıcıda yok, kod çözücüde nedensel | çeviri, boşluk doldurma | orijinal Transformer, T5 (M4.14) |

GPT, orijinal kod çözücü bloğundan kodlayıcıya bakan çapraz dikkat alt katmanını atar: blok başına iki alt katman kalır (maskeli öz-dikkat + MLP). Kalan her şey bu modülün konusudur.

### Dil modelleme hedefi {#hedef}

Bir **dil modeli** token dizileri üzerinde bir olasılık dağılımıdır. Olasılığın zincir kuralı **hiçbir varsayım olmadan** her ortak dağılımı koşullu olasılıkların çarpımına ayırır:

$$p(x_1,\dots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_1,\dots,x_{t-1}).$$

**Otoregresif** model her koşullu dağılımı ayrı ayrı modeller: önek verildiğinde **bir sonraki token** üzerinde bir dağılım. (Yaklaşıklık, eşitlikte değil modelde: gerçek koşullu dağılımı sonlu bir ağ ve sonlu bir bağlam penceresi $T$ ile tahmin ediyoruz.)

:::türetim Kayıp: dizinin negatif log-olabilirliği
Parametreler $\theta$, eğitim dizisi $x_{1:T}$. En çok olabilirlik (M1.4) $\log p_\theta(x_{1:T})$'yi büyütür; zincir kuralıyla log çarpımı toplama çevirir:
$$-\log p_\theta(x_{1:T})=\sum_{t=1}^{T}-\log p_\theta(x_t\mid x_{<t}).$$
Her terim, $t$. konumdaki tahminin (sözlük üzerinde softmax) **gerçek token'a** karşı çapraz entropisidir. Eğitim kaybı bunların konum ve batch üzerindeki ortalamasıdır:
$$\mathcal L=\frac1{BT}\sum_{b,t}-\log\operatorname{softmax}(\mathbf z_{b,t})_{x_{b,t+1}}.$$
Bu yüzden PyTorch'ta tek satırdır: `F.cross_entropy(logits.view(-1, V), targets.view(-1))`, hedef de girdinin bir konum kaydırılmışı.
:::

<figure class="fig wide"><svg viewBox="0 0 640 210" role="img" aria-label="Girdi dizisi bir konum kaydırılarak hedef dizisi elde edilir; nedensel maske alt üçgen dışındaki skorları eksi sonsuz yapar"><defs><marker id="g3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><text x="20" y="22" text-anchor="start" class="sm">girdi</text><text x="20" y="112" text-anchor="start" class="sm">hedef</text><rect x="80" y="8" width="70" height="24" rx="5" class="soft-a ca"/><text x="115" y="24" text-anchor="middle" class="sm">&lt;s&gt;</text><rect x="80" y="98" width="70" height="24" rx="5" class="soft-p cp"/><text x="115" y="114" text-anchor="middle" class="sm">Kedi</text><line x1="115" y1="34" x2="115" y2="96" class="ink" marker-end="url(#g3)"/><rect x="160" y="8" width="70" height="24" rx="5" class="soft-a ca"/><text x="195" y="24" text-anchor="middle" class="sm">Kedi</text><rect x="160" y="98" width="70" height="24" rx="5" class="soft-p cp"/><text x="195" y="114" text-anchor="middle" class="sm">masanın</text><line x1="195" y1="34" x2="195" y2="96" class="ink" marker-end="url(#g3)"/><rect x="240" y="8" width="70" height="24" rx="5" class="soft-a ca"/><text x="275" y="24" text-anchor="middle" class="sm">masanın</text><rect x="240" y="98" width="70" height="24" rx="5" class="soft-p cp"/><text x="275" y="114" text-anchor="middle" class="sm">üstüne</text><line x1="275" y1="34" x2="275" y2="96" class="ink" marker-end="url(#g3)"/><rect x="320" y="8" width="70" height="24" rx="5" class="soft-a ca"/><text x="355" y="24" text-anchor="middle" class="sm">üstüne</text><rect x="320" y="98" width="70" height="24" rx="5" class="soft-p cp"/><text x="355" y="114" text-anchor="middle" class="sm">atladı</text><line x1="355" y1="34" x2="355" y2="96" class="ink" marker-end="url(#g3)"/><text x="80" y="150" text-anchor="start" class="sm">Bir dizi T tahmin verir: her konum yalnız kendisi ve öncesini görür.</text><text x="80" y="168" text-anchor="start" class="sm">Kayıp = T konumun çapraz entropisinin ortalaması (öğretmen zorlaması).</text><text x="491" y="48" transform="rotate(-40 491 48)" class="sm" font-size="10">&lt;s&gt;</text><text x="525" y="48" transform="rotate(-40 525 48)" class="sm" font-size="10">Kedi</text><text x="559" y="48" transform="rotate(-40 559 48)" class="sm" font-size="10">masanın</text><text x="593" y="48" transform="rotate(-40 593 48)" class="sm" font-size="10">üstüne</text><text x="472" y="72" text-anchor="end" class="sm" font-size="10">&lt;s&gt;</text><rect x="478" y="52" width="32" height="32" class="soft-a" stroke-width="0.5"/><text x="494" y="72" text-anchor="middle" class="sm" font-size="10"></text><rect x="512" y="52" width="32" height="32" class="mute" stroke-width="0.5"/><text x="528" y="72" text-anchor="middle" class="sm" font-size="10">−∞</text><rect x="546" y="52" width="32" height="32" class="mute" stroke-width="0.5"/><text x="562" y="72" text-anchor="middle" class="sm" font-size="10">−∞</text><rect x="580" y="52" width="32" height="32" class="mute" stroke-width="0.5"/><text x="596" y="72" text-anchor="middle" class="sm" font-size="10">−∞</text><text x="472" y="106" text-anchor="end" class="sm" font-size="10">Kedi</text><rect x="478" y="86" width="32" height="32" class="soft-a" stroke-width="0.5"/><text x="494" y="106" text-anchor="middle" class="sm" font-size="10"></text><rect x="512" y="86" width="32" height="32" class="soft-a" stroke-width="0.5"/><text x="528" y="106" text-anchor="middle" class="sm" font-size="10"></text><rect x="546" y="86" width="32" height="32" class="mute" stroke-width="0.5"/><text x="562" y="106" text-anchor="middle" class="sm" font-size="10">−∞</text><rect x="580" y="86" width="32" height="32" class="mute" stroke-width="0.5"/><text x="596" y="106" text-anchor="middle" class="sm" font-size="10">−∞</text><text x="472" y="140" text-anchor="end" class="sm" font-size="10">masanın</text><rect x="478" y="120" width="32" height="32" class="soft-a" stroke-width="0.5"/><text x="494" y="140" text-anchor="middle" class="sm" font-size="10"></text><rect x="512" y="120" width="32" height="32" class="soft-a" stroke-width="0.5"/><text x="528" y="140" text-anchor="middle" class="sm" font-size="10"></text><rect x="546" y="120" width="32" height="32" class="soft-a" stroke-width="0.5"/><text x="562" y="140" text-anchor="middle" class="sm" font-size="10"></text><rect x="580" y="120" width="32" height="32" class="mute" stroke-width="0.5"/><text x="596" y="140" text-anchor="middle" class="sm" font-size="10">−∞</text><text x="472" y="174" text-anchor="end" class="sm" font-size="10">üstüne</text><rect x="478" y="154" width="32" height="32" class="soft-a" stroke-width="0.5"/><text x="494" y="174" text-anchor="middle" class="sm" font-size="10"></text><rect x="512" y="154" width="32" height="32" class="soft-a" stroke-width="0.5"/><text x="528" y="174" text-anchor="middle" class="sm" font-size="10"></text><rect x="546" y="154" width="32" height="32" class="soft-a" stroke-width="0.5"/><text x="562" y="174" text-anchor="middle" class="sm" font-size="10"></text><rect x="580" y="154" width="32" height="32" class="soft-a" stroke-width="0.5"/><text x="596" y="174" text-anchor="middle" class="sm" font-size="10"></text><text x="544" y="204" text-anchor="middle" class="sm" font-size="10">satır: sorgu, sütun: anahtar</text></svg><figcaption>Şekil 3 — Dil modeli eğitimi. Hedef, girdinin bir konum sola kaydırılmışıdır; her konum bir sonraki token'ı tahmin eder. Nedensel maske (sağ) j &gt; i skorlarını −∞ yapar, softmax'tan sonra bu ağırlıklar tam 0 olur: konum i, i'den sonraki girdileri göremez. Böylece tek bir ileri geçişte T ayrı tahmin, sızıntısız ve paralel eğitilir.</figcaption></figure>

**Öğretmen zorlaması** (teacher forcing): eğitimde $t$. konum, modelin kendi önceki tahminlerini değil **gerçek** öneki görür. Bu, nedensel maske sayesinde tek bir ileri geçişte $T$ tahminin **birlikte** yapılabilmesini sağlar; RNN'in adım adım işlemesine göre en büyük hız kazancı budur. Dersteki "yalnız tek bir (o anki girdinin) gömmesi son softmax'a ulaşır" cümlesi **çıkarım** için doğrudur: üretimde yalnız son konumun logit'i kullanılır. Eğitimde her konumun logit'i kayba girer.

:::kanıt Nedensel maske gerçekten sızıntıyı önler
İddia: maskeli bir yığının $t$. konumdaki çıktısı yalnız $x_{1:t}$'ye bağlıdır. Tümevarım: gömme katmanı ve MLP/LN konum konum çalışır (zamana dağıtılmış), dolayısıyla konum $t$'nin çıktısı yalnız konum $t$'nin girdisine bağlıdır. Maskeli dikkatte konum $t$'nin çıktısı $\sum_{j}\alpha_{tj}\mathbf v_j$; $j>t$ için skor $-\infty$, $\alpha_{tj}=e^{-\infty}/Z=0$ **tam olarak**. Kalan $\mathbf v_j$ ($j\le t$) bir önceki katmanın $j\le t$ konumlarından gelir; tümevarım varsayımıyla onlar da yalnız $x_{1:j}\subseteq x_{1:t}$'ye bağlıdır. Artık toplama konum konumdur. Dolayısıyla her katmanda iddia korunur. ∎

Koşul: maske softmax'tan **önce** ve $-\infty$ (ya da dtype'ın en küçük sayısı) ile uygulanmalı. Skoru 0 ile çarpmak yanlıştır: $e^0=1$, gelecek token pay alır (pratik kartı P5).
:::

**Perpleksite.** Ortalama negatif log-olabilirliğin üstelidir: $\mathrm{PPL}=e^{\mathcal L}$. Yorum: model her adımda ortalama kaç eşit olası seçenek arasında kalmış gibidir. Düzgün dağılım $\mathrm{PPL}=V$ verir. **Pratik kontrol:** rastgele başlatılmış bir dil modelinin ilk kaybı $\approx\ln V$ olmalıdır (GPT-2'de $\ln 50257\approx10.82$). Belirgin biçimde büyükse logit'ler başlangıçta fazla büyüktür (başlatma ölçeği ya da bağlı çıkış katmanının ölçeği hatalı).

:::yanılgı "Model eğitimde de kelime kelime üretir."
Hayır. Eğitimde $T$ konumun hepsi paralel hesaplanır ve maske her konumun yalnız geçmişi görmesini sağlar. Kelime kelime üretim yalnız çıkarımda gerekir; çünkü o zaman gelecekteki token'lar henüz yoktur.
:::

### GPT'nin yapısı {#yapi}

<figure class="fig wide"><svg viewBox="0 0 640 510" role="img" aria-label="GPT modelinin aşağıdan yukarı yapısı: token ve konum gömmeleri toplanır, L kez tekrarlanan bloklarda artık akışa önce dikkat sonra MLP dalı eklenir, son LayerNorm ve bağlı çıkış katmanı logit üretir"><defs><marker id="g1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="110" y="476" width="120" height="24" rx="5" class="mute"/><text x="170" y="492" text-anchor="middle" class="sm">token kimlikleri (T)</text><rect x="40" y="428" width="120" height="26" rx="5" class="soft-a ca"/><text x="100" y="445" text-anchor="middle" class="sm">wte[idx] (V × d)</text><rect x="180" y="428" width="120" height="26" rx="5" class="soft-a ca"/><text x="240" y="445" text-anchor="middle" class="sm">wpe[0..T) (T × d)</text><line x1="150" y1="476" x2="110" y2="456" class="ink" marker-end="url(#g1)"/><line x1="190" y1="476" x2="230" y2="456" class="ink" marker-end="url(#g1)"/><circle cx="170" cy="408" r="9" class="ink" fill="none"/><text x="170" y="412" text-anchor="middle" class="sm">+</text><line x1="100" y1="428" x2="162" y2="412" class="ink" marker-end="url(#g1)"/><line x1="240" y1="428" x2="178" y2="412" class="ink" marker-end="url(#g1)"/><rect x="20" y="160" width="330" height="234" rx="8" class="ink dash" fill="none"/><text x="340" y="178" text-anchor="end" class="sm">× L blok</text><line x1="170" y1="399" x2="170" y2="299" class="ink"/><circle cx="170" cy="290" r="9" class="ink" fill="none"/><text x="170" y="294" text-anchor="middle" class="sm">+</text><line x1="170" y1="281" x2="170" y2="194" class="ink"/><circle cx="170" cy="185" r="9" class="ink" fill="none"/><text x="170" y="189" text-anchor="middle" class="sm">+</text><line x1="170" y1="176" x2="170" y2="123" class="ink" marker-end="url(#g1)"/><line x1="170" y1="375" x2="250" y2="375" class="ink"/><line x1="250" y1="375" x2="250" y2="366" class="ink" marker-end="url(#g1)"/><rect x="220" y="340" width="60" height="24" rx="5" class="soft-b cb"/><text x="250" y="356" text-anchor="middle" class="sm">LN<tspan baseline-shift="sub" font-size="9">1</tspan></text><line x1="250" y1="340" x2="250" y2="332" class="ink" marker-end="url(#g1)"/><rect x="190" y="302" width="120" height="28" rx="5" class="soft-p cp"/><text x="250" y="320" text-anchor="middle" class="sm">nedensel MHA</text><line x1="250" y1="302" x2="250" y2="290" class="ink"/><line x1="250" y1="290" x2="180" y2="290" class="ink" marker-end="url(#g1)"/><line x1="170" y1="270" x2="90" y2="270" class="ink"/><line x1="90" y1="270" x2="90" y2="262" class="ink" marker-end="url(#g1)"/><rect x="60" y="236" width="60" height="24" rx="5" class="soft-b cb"/><text x="90" y="252" text-anchor="middle" class="sm">LN<tspan baseline-shift="sub" font-size="9">2</tspan></text><line x1="90" y1="236" x2="90" y2="228" class="ink" marker-end="url(#g1)"/><rect x="30" y="198" width="120" height="28" rx="5" class="soft-p cp"/><text x="90" y="216" text-anchor="middle" class="sm">MLP d→4d→d</text><line x1="90" y1="198" x2="90" y2="185" class="ink"/><line x1="90" y1="185" x2="160" y2="185" class="ink" marker-end="url(#g1)"/><text x="178" y="240" text-anchor="start" class="sm ta">artık akış</text><rect x="130" y="95" width="80" height="26" rx="5" class="soft-b cb"/><text x="170" y="112" text-anchor="middle" class="sm">LN<tspan baseline-shift="sub" font-size="9">f</tspan></text><rect x="100" y="50" width="140" height="28" rx="5" class="soft-a ca"/><text x="170" y="68" text-anchor="middle" class="sm">· wteᵀ → logit (T × V)</text><line x1="170" y1="95" x2="170" y2="80" class="ink" marker-end="url(#g1)"/><rect x="110" y="8" width="120" height="26" rx="5" class="mute"/><text x="170" y="25" text-anchor="middle" class="sm">softmax → p(x<tspan baseline-shift="sub" font-size="9">t+1</tspan>)</text><line x1="170" y1="50" x2="170" y2="36" class="ink" marker-end="url(#g1)"/><line x1="40" y1="454" x2="40" y2="492" class="ca dash"/><line x1="40" y1="492" x2="10" y2="492" class="ca dash"/><line x1="10" y1="492" x2="10" y2="64" class="ca dash"/><line x1="10" y1="64" x2="98" y2="64" class="ca dash" marker-end="url(#g1)"/><text transform="rotate(-90 24 470)" x="24" y="470" text-anchor="start" class="sm ta">aynı matris</text><text x="380" y="200" text-anchor="start" class="sm">• Her blok artık akışı okur (LN ile normalize</text><text x="380" y="220" text-anchor="start" class="sm">  edilmiş bir kopyasını) ve sonucunu ona ekler.</text><text x="380" y="240" text-anchor="start" class="sm">• Dikkat: token'lar arası bilgi alışverişi</text><text x="380" y="260" text-anchor="start" class="sm">  (yalnız geçmişe, nedensel maske).</text><text x="380" y="280" text-anchor="start" class="sm">• MLP: her token'da ayrı ayrı, aynı ağırlıklar</text><text x="380" y="300" text-anchor="start" class="sm">  ("time-distributed").</text><text x="380" y="320" text-anchor="start" class="sm">• Konum bilgisi yalnız wpe'den gelir.</text><text x="380" y="340" text-anchor="start" class="sm">• Çıkış katmanı wte'nin devriği (weight tying).</text></svg><figcaption>Şekil 1 — GPT (Pre-LN, GPT-2 düzeni). Dikey çizgi artık akıştır (residual stream): her blok onu okur, iki dalda (nedensel çok başlı dikkat, MLP) hesap yapar ve sonuçları akışa ekler. En üstte son LayerNorm ve gömme matrisinin devriğiyle çarpım her konum için sözlük üzerinde logit üretir; kesikli çizgi aynı matrisin iki kez kullanıldığını gösterir.</figcaption></figure>

Batch $B$, bağlam $T$, model boyu $d$, baş sayısı $h$, sözlük $V$. Shape'leri adım adım izle:

| adım | işlem | shape |
|---|---|---|
| girdi | token kimlikleri `idx` | $(B,T)$ tamsayı |
| gömme | $\mathbf x^0=W_{te}[\text{idx}]+W_{pe}[0..T)$ | $(B,T,d)$ |
| blok $l$, dal 1 | $\mathbf x\leftarrow\mathbf x+\mathrm{MHA}_{\text{nedensel}}(\mathrm{LN}_1(\mathbf x))$ | $(B,T,d)$ |
| — içinde | $\mathbf Q,\mathbf K,\mathbf V$: $(B,h,T,d/h)$; skorlar $(B,h,T,T)$ | |
| blok $l$, dal 2 | $\mathbf x\leftarrow\mathbf x+\mathrm{MLP}(\mathrm{LN}_2(\mathbf x))$ | $(B,T,d)$ |
| — içinde | $\mathrm{MLP}(\mathbf u)=W_2\,\mathrm{GELU}(W_1\mathbf u+\mathbf b_1)+\mathbf b_2$, ara katman $4d$ | $(B,T,4d)$ |
| son | $\mathrm{LN}_f(\mathbf x^L)\,W_{te}^\top$ | $(B,T,V)$ logit |

Gözlemler:

- Her blok **aynı shape'i** korur, bu yüzden istediğin kadar yığabilirsin; artık bağlantılar da bunu gerektirir (girdi ve çıktı boyutu eşit olmalı).
- **Token'lar arası bilgi alışverişinin tek yeri dikkattir.** LN, MLP, gömme, çıkış katmanı konum konum çalışır. Dikkat "iletişim", MLP "hesap" olarak düşünülebilir (Karpathy'nin benzetmesi).
- **Konum bilgisinin tek kaynağı $W_{pe}$'dir.** Dikkat permütasyona eş-değişkendir (girdi token'larını karıştırırsan çıktılar aynı biçimde karışır); konum gömmesi olmadan model "kedi köpeği kovaladı" ile "köpek kediyi kovaladı"yı (kelime sırası dışında aynı token'lar) ayırt edemezdi. GPT-1/2/3 **öğrenilen mutlak** konum gömmeleri kullanır: $T\times d$'lik bir tablo; bu yüzden $T$'den uzun dizileri hiç göremez. Orijinal Transformer'ın sinüs/kosinüs kodlaması ve bugünkü RoPE M4.14'te.

:::sezgi Artık akış (residual stream)
Pre-LN'de blokların hiçbiri $\mathbf x$'in üzerine yazmaz; yalnız ekler. Açarsak:
$$\mathbf x^{L}=\mathbf x^0+\sum_{l=1}^{L}\Big(\mathrm{MHA}_l(\cdot)+\mathrm{MLP}_l(\cdot)\Big).$$
Akış, $d$ boyutlu ortak bir "çalışma belleği"dir: gömme bir token'ın kimliğini yazar, dikkat başları başka konumlardan bilgi kopyalar, MLP'ler o bilgi üzerinde hesap yapıp sonucu ekler, en sonda çıkış katmanı akıştan bir sonraki token'ı okur. Yorumlanabilirlik çalışmaları (Elhage ve ark. 2021, "A Mathematical Framework for Transformer Circuits") bu bakışla başlar; "logit lens" ara katmanlardaki akışı doğrudan $\mathrm{LN}_f(\cdot)W_{te}^\top$'den geçirip modelin o katmandaki tahminini okur. Bunlar **yorum araçlarıdır**, modelin tanımı değil.
:::

### MLP alt katmanı {#mlp}

Orijinal makaledeki "konum bazlı ileri beslemeli ağ" (position-wise feed-forward) her token'a ayrı ayrı ama **aynı ağırlıklarla** uygulanan iki katmanlı bir ağdır:
$$\mathrm{FFN}(\mathbf u)=\max(0,\mathbf u W_1+\mathbf b_1)W_2+\mathbf b_2,\qquad W_1\in\mathbb R^{d\times d_{ff}},\;W_2\in\mathbb R^{d_{ff}\times d},$$
makalede $d=512$, $d_{ff}=2048=4d$. Aynı işlem, çekirdek boyu 1 olan iki konvolüsyon olarak da görülebilir.

GPT-1'den itibaren ReLU yerine **GELU** kullanılır: $\mathrm{GELU}(z)=z\,\Phi(z)$ ($\Phi$: standart normal dağılım fonksiyonu). GPT-2 ve nanoGPT tanh yaklaşımını kullanır:
$$\mathrm{GELU}(z)\approx\tfrac12z\Big(1+\tanh\big(\sqrt{2/\pi}\,(z+0.044715z^3)\big)\Big).$$
Bu bir **yaklaşıklıktır** (en büyük mutlak hata $\approx4.7\cdot10^{-4}$); ağırlık yüklerken modelin hangisiyle eğitildiğine uymak gerekir: PyTorch'ta `nn.GELU(approximate="tanh")`.

:::sezgi MLP bir anahtar–değer belleği gibi okunabilir
$W_1$'in her sütunu $\mathbf k_i$ ve $W_2$'nin her satırı $\mathbf v_i$ olsun: $\mathrm{FFN}(\mathbf u)=\sum_{i=1}^{d_{ff}}f(\mathbf u\cdot\mathbf k_i)\,\mathbf v_i$. Bu, softmax yerine eleman bazlı bir aktivasyon kullanan dikkate benzer: girdi "anahtarlarla" eşleştirilir, eşleşme gücü oranında "değerler" toplanır. Geva ve ark. (2021) eğitilmiş modellerde bazı anahtarların belirli metin kalıplarına tepki verdiğini, değerlerin de belirli sonraki token'ları güçlendirdiğini göstermiştir. Parametrelerin üçte ikisi MLP'dedir (aşağıdaki sayım); "olgusal bilginin" büyük kısmının burada saklandığı yönünde bulgular var — ama bu bir yorumdur, kesin bir ayrım değil.
:::

**Modern değişiklik: SwiGLU.** LLaMA ve sonrası $\big(\mathrm{SiLU}(\mathbf uW_1)\odot\mathbf uW_3\big)W_2$ kullanır: üç matris. Parametre sayısını sabit tutmak için $d_{ff}=\tfrac83d$ seçilir: $3\cdot d\cdot\tfrac83d=8d^2$, iki matrisli $4d$'li MLP'ninkiyle aynı. Shazeer (2020) aynı bütçede biraz daha düşük kayıp bildirir.

### LayerNorm ve yeri {#ln}

LayerNorm (M3.6) her token'ın $d$ boyutlu vektörünü **kendi** bileşenleri üzerinden normalize eder (batch'ten bağımsız, bu yüzden değişken uzunluklu dizilerde ve tek örnekli çıkarımda sorunsuzdur):
$$\mathrm{LN}(\mathbf x)=\boldsymbol\gamma\odot\frac{\mathbf x-\mu}{\sqrt{\sigma^2+\epsilon}}+\boldsymbol\beta,\qquad \mu=\frac1d\sum_ix_i,\;\sigma^2=\frac1d\sum_i(x_i-\mu)^2.$$

:::uyarı Kaynak düzeltmesi: LayerNorm'un öğrenilen parametreleri
Ders slaytında "LayerNorm iki öğrenilebilir parametre ekler: bir ortalama ve bir standart sapma" yazıyor. Doğrusu: ortalama $\mu$ ve standart sapma $\sigma$ **öğrenilmez**, her token için o anki vektörden **hesaplanır**. Öğrenilen parametreler bir **ölçek** $\boldsymbol\gamma$ ve bir **kaydırma** $\boldsymbol\beta$'dır ve skaler değil, **$d$ boyutlu vektörlerdir** (özellik başına bir tane): bir LayerNorm $2d$ parametre ekler. GPT-2 small'da her LN 1536, 25 LN toplam 38 400 parametre.
:::

<figure class="fig wide"><svg viewBox="0 0 640 370" role="img" aria-label="Solda Post-LN: artık toplamdan sonra LayerNorm. Sağda Pre-LN: LayerNorm yalnız dalın girdisine uygulanır, artık yol normalize edilmez"><defs><marker id="g2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><text x="160" y="20" text-anchor="middle" class="sm">Post-LN (Vaswani ve ark. 2017)</text><text x="100" y="268" text-anchor="middle" class="sm">xˡ</text><line x1="100" y1="256" x2="100" y2="160" class="ink"/><line x1="100" y1="230" x2="200" y2="230" class="ink"/><line x1="200" y1="230" x2="200" y2="202" class="ink" marker-end="url(#g2)"/><rect x="150" y="170" width="100" height="30" rx="5" class="soft-p cp"/><text x="200" y="189" text-anchor="middle" class="sm">F (dikkat/MLP)</text><line x1="200" y1="170" x2="200" y2="151" class="ink"/><line x1="200" y1="151" x2="110" y2="151" class="ink" marker-end="url(#g2)"/><circle cx="100" cy="151" r="9" class="ink" fill="none"/><text x="100" y="155" text-anchor="middle" class="sm">+</text><line x1="100" y1="142" x2="100" y2="112" class="ink" marker-end="url(#g2)"/><rect x="70" y="82" width="60" height="28" rx="5" class="soft-b cb"/><text x="100" y="100" text-anchor="middle" class="sm">LN</text><line x1="100" y1="82" x2="100" y2="50" class="ink" marker-end="url(#g2)"/><text x="100" y="42" text-anchor="middle" class="sm">xˡ⁺¹</text><text x="40" y="300" text-anchor="start" class="sm">xˡ⁺¹ = LN(xˡ + F(xˡ))</text><text x="40" y="320" text-anchor="start" class="sm">Özdeşlik yolu her blokta LN'den geçer.</text><text x="40" y="338" text-anchor="start" class="sm">Isınma (warmup) olmadan büyük öğrenme</text><text x="40" y="356" text-anchor="start" class="sm">oranında kararsız (Xiong ve ark. 2020).</text><text x="470" y="20" text-anchor="middle" class="sm">Pre-LN (GPT-2, 2019 ve sonrası)</text><text x="410" y="268" text-anchor="middle" class="sm">xˡ</text><line x1="410" y1="256" x2="410" y2="150" class="ink"/><line x1="410" y1="230" x2="510" y2="230" class="ink"/><line x1="510" y1="230" x2="510" y2="222" class="ink" marker-end="url(#g2)"/><rect x="480" y="194" width="60" height="26" rx="5" class="soft-b cb"/><text x="510" y="211" text-anchor="middle" class="sm">LN</text><line x1="510" y1="194" x2="510" y2="182" class="ink" marker-end="url(#g2)"/><rect x="460" y="150" width="100" height="30" rx="5" class="soft-p cp"/><text x="510" y="169" text-anchor="middle" class="sm">F (dikkat/MLP)</text><line x1="510" y1="150" x2="510" y2="131" class="ink"/><line x1="510" y1="131" x2="420" y2="131" class="ink" marker-end="url(#g2)"/><line x1="410" y1="150" x2="410" y2="140" class="ink"/><circle cx="410" cy="131" r="9" class="ink" fill="none"/><text x="410" y="135" text-anchor="middle" class="sm">+</text><line x1="410" y1="122" x2="410" y2="50" class="ink" marker-end="url(#g2)"/><text x="410" y="42" text-anchor="middle" class="sm">xˡ⁺¹</text><text x="350" y="300" text-anchor="start" class="sm">xˡ⁺¹ = xˡ + F(LN(xˡ))</text><text x="350" y="320" text-anchor="start" class="sm">Özdeşlik yolu temiz: xᴸ = x⁰ + Σ Fˡ.</text><text x="350" y="338" text-anchor="start" class="sm">Akış büyür (≈ √L); en sonda LN<tspan baseline-shift="sub" font-size="9">f</tspan> gerekir.</text><text x="350" y="356" text-anchor="start" class="sm">Isınmaya daha az bağımlı, derinde kararlı.</text></svg><figcaption>Şekil 2 — LayerNorm'un iki yerleşimi. Orijinal Transformer'da (ve Géron/ders slaytlarındaki "Add &amp; Norm" kutusunda) normalizasyon toplamdan sonra gelir. GPT-2'den beri neredeyse tüm büyük dil modelleri LN'i dalın girişine taşır; artık yol bloklar boyunca hiç dokunulmadan akar. Deney l3 iki düzenin başlangıçtaki gradyan ve akış ölçeğini karşılaştırıyor.</figcaption></figure>

Orijinal Transformer (ve slaytlardaki "Add & Norm" kutusu) **Post-LN**'dir: $\mathbf x^{l+1}=\mathrm{LN}(\mathbf x^l+F(\mathbf x^l))$. GPT-2 (2019) normalizasyonu dalın **girişine** taşıdı ve en sona bir LN ekledi: **Pre-LN**, $\mathbf x^{l+1}=\mathbf x^l+F(\mathrm{LN}(\mathbf x^l))$. Bugün neredeyse bütün büyük dil modelleri Pre-LN'dir (çoğu LN yerine RMSNorm ile).

:::türetim Pre-LN'de özdeşlik yolu
Pre-LN'de $\mathbf x^{L}=\mathbf x^l+\sum_{k=l}^{L-1}F_k(\mathrm{LN}(\mathbf x^k))$. Zincir kuralı (Jacobian'lar):
$$\frac{\partial\mathbf x^L}{\partial\mathbf x^l}=I+\sum_{k=l}^{L-1}\frac{\partial F_k(\mathrm{LN}(\mathbf x^k))}{\partial\mathbf x^l}.$$
İlk terim **özdeşliktir**: kayıptan gelen gradyan $\partial\mathcal L/\partial\mathbf x^L$, hiçbir katmanla çarpılmadan her bloğun girişine ulaşır (ResNet'teki türetimin aynısı, M4.2). Post-LN'de ise her blokta $\mathbf x^{l+1}=\mathrm{LN}(\cdot)$ olduğundan
$$\frac{\partial\mathbf x^{l+1}}{\partial\mathbf x^l}=J_{\mathrm{LN}}\Big(I+\frac{\partial F_l}{\partial\mathbf x^l}\Big),$$
yani özdeşlik yolu da her blokta LN'in Jacobian'ı ile çarpılır; $L$ blok boyunca bu çarpımlar birikir. (Bu, gradyanın "kaybolacağı" anlamına gelmez; yolun temiz olmadığını söyler. Ne olacağı Jacobian'ların değerine bağlıdır.)
:::

**Ölçülen fark.** Xiong ve ark. (2020) başlangıçta Post-LN'de **çıkışa yakın** katmanların gradyanının derinlikten bağımsız olarak büyük kaldığını, Pre-LN'de ise $\sim1/\sqrt L$ ile küçüldüğünü gösterdi; büyük öğrenme oranıyla birleşince Post-LN'in ilk adımlarda ıraksadığını ve orijinal Transformer'ın bu yüzden **öğrenme oranı ısınmasına (warmup)** muhtaç olduğunu buldular. Pre-LN ısınmasız da eğitilebilir. Bedeli: Pre-LN'de akış normalize edilmediği için **büyür** (varyanslar toplanır, $\approx\sqrt{L}$); son bir $\mathrm{LN}_f$ gerekir. Aşağıdaki deney bu iki olguyu (gradyan profili, akış büyümesi) rastgele başlatılmış bir yığında ölçüyor.

::lab l3

:::ayrıntı Post-LN tamamen terk edilmedi
Liu ve ark. (2020) uygun başlatmayla Post-LN'in de ısınmasız eğitilebildiğini ve iyi eğitildiğinde biraz daha iyi sonuç verebildiğini gösterdi (Admin başlatması); DeepNet (Wang ve ark. 2022) 1000 katmanlı Post-LN Transformer'ı artık bağlantıyı büyütüp dalların başlatmasını küçülterek (DeepNorm) eğitti. Pre-LN'in derin katmanlarının "az iş yaptığı" (akış büyüdükçe her bloğun göreli katkısı küçülür) da tartışılan bir eleştiridir.
:::

### Parametre ve hesap sayımı {#sayim}

:::türetim Blok başına $12d^2+13d$
Bias'lı doğrusal katman $d_{\text{in}}\to d_{\text{out}}$: $d_{\text{in}}d_{\text{out}}+d_{\text{out}}$ parametre.
- **Dikkat:** Q, K, V izdüşümleri (tek bir $d\to3d$ katmanı olarak da yazılır) $3(d^2+d)$; çıkış izdüşümü $W_O$: $d^2+d$. Toplam $4d^2+4d$. **Baş sayısı $h$ parametreyi değiştirmez**: her baş $d/h$ boyutlu bir dilim kullanır.
- **MLP:** $d\to4d$: $4d^2+4d$; $4d\to d$: $4d^2+d$. Toplam $8d^2+5d$.
- **İki LayerNorm:** $2\cdot2d=4d$.

Toplam $12d^2+13d$. $d=768$: $12\cdot589\,824+13\cdot768=7\,077\,888+9\,984=7\,087\,872$.

**Model:** $Vd$ (token gömme) $+\,Td$ (konum gömme) $+\,L(12d^2+13d)$ $+\,2d$ (son LN), çıkış katmanı bağlıysa ek parametre yok. GPT-2 small ($V=50\,257$, $T=1024$, $d=768$, $L=12$):
$$38\,597\,376+786\,432+85\,054\,464+1\,536=124\,439\,808.$$
:::

Aynı formülle GPT ailesi (hepsi $d_{ff}=4d$, bağlı çıkış):

| model | $L$ | $d$ | baş $h$ | $V$ | $T$ | formül | yayımlanan |
|---|---|---|---|---|---|---|---|
| GPT-1 | 12 | 768 | 12 | 40 478 | 512 | 116.5M | "117M" |
| GPT-2 small | 12 | 768 | 12 | 50 257 | 1024 | 124.4M | "117M" (ilk), sonra 124M |
| GPT-2 medium | 24 | 1024 | 16 | 50 257 | 1024 | 354.8M | "345M" |
| GPT-2 large | 36 | 1280 | 20 | 50 257 | 1024 | 774.0M | "762M" |
| GPT-2 XL | 48 | 1600 | 25 | 50 257 | 1024 | 1 557.6M | "1542M" |
| GPT-3 | 96 | 12 288 | 96 | 50 257 | 2048 | 174.6B | "175B" |

GPT-2 makalesinin tablosundaki sayılar formülden biraz küçüktür (makale küçük modeli de "117M" diye vermişti); farkın bir kısmı sayım hatasıdır, OpenAI daha sonra küçük modelin sayısını 124M olarak düzeltti. Yayımlanan ağırlıkları yükleyip saydığında formülün sayısını bulursun (ödev 1).

**Hesap.** Bir parametreli matris çarpımında her ağırlık, her token için **bir çarpma ve bir toplama** yapar: 2 FLOP. Gömme araması çarpım değildir; çıkış izdüşümü ise $Vd$ ağırlıkla çarpım yapar. Bu yüzden token başına ileri geçiş $\approx2N$ FLOP'tur ($N$: gömme dışı parametre + çıkış izdüşümü).

:::türetim Dikkatin FLOP'u: $2LTd$
$t$. konumdaki sorgu $t$ anahtarla iç çarpım yapar ($t\cdot d$ çarpma–toplama, tüm başlar birlikte) ve $t$ değerin ağırlıklı toplamını alır (yine $t\cdot d$): $4td$ FLOP. Nedensel dizide $t=1..T$ ortalaması $\approx T/2$, katman başına token başına $\approx 2Td$; $L$ katmanda $2LTd$ (Kaplan ve ark. 2020, Tablo 1). Oran:
$$\frac{2LTd}{2\cdot12Ld^2}=\frac{T}{12d}.$$
GPT-2 small, $T=1024$: $1024/9216\approx11\%$. Bağlam $T\approx12d$'yi geçince dikkat hesabı parametreli çarpımları geçer.
:::

**Eğitim ≈ 6N FLOP/token.** Geri geçiş her doğrusal katman için iki matris çarpımı yapar: aktivasyonlara göre gradyan ($\partial\mathcal L/\partial\mathbf u=\boldsymbol\delta W^\top$) ve ağırlıklara göre gradyan ($\partial\mathcal L/\partial W=\mathbf u^\top\boldsymbol\delta$); her biri ileri çarpımla aynı boyutta. İleri $2N$ + geri $4N$ = $6N$. GPT-3: $6\times175\cdot10^9\times300\cdot10^9$ token $\approx3.15\cdot10^{23}$ FLOP (makalede $3.14\cdot10^{23}$).

::lab l1

### Ağırlık bağlama ve başlatma {#baglama}

**Ağırlık bağlama** (weight tying; Press & Wolf 2017, Inan ve ark. 2017): çıkış katmanı ayrı bir $d\times V$ matris değil, token gömme matrisinin devriğidir: $\text{logit}_v=\mathbf h\cdot W_{te}[v]$. Gerekçe: girişte benzer kullanılan token'lar benzer gömmeler alır; çıkışta da benzer bağlamlarda tahmin edilmeleri gerekir — iki matris aynı geometriyi öğrenmeye çalışır. Kazanç: GPT-2 small'da $Vd=38.6$M parametre (toplamın %31'i); ayrıca nadir token'ların gömmeleri de çıkış katmanından gelen her gradyanla güncellenir. PyTorch'ta tek satır: `self.head.weight = self.wte.weight` (aynı `Parameter` nesnesi; gradyanlar iki kullanımdan toplanır). Büyük modellerde (gömme payı küçük) bağlama çoğu kez kullanılmaz: LLaMA ayrı matris kullanır.

**Başlatma.** GPT-2/nanoGPT tüm doğrusal ve gömme ağırlıklarını $\mathcal N(0,0.02^2)$ ile başlatır, bias'ları 0 yapar.

:::türetim Artık dalların $1/\sqrt{N}$ ölçeklemesi
Pre-LN'de $\mathbf x^L=\mathbf x^0+\sum_{k=1}^{N}\mathbf f_k$, $N=2L$ dal (her blokta dikkat ve MLP). Başlangıçta dal çıktıları yaklaşık ilişkisiz, her birinin bileşen varyansı $s^2$ ise
$$\mathrm{Var}(x^L_i)\approx\mathrm{Var}(x^0_i)+Ns^2.$$
$N$ büyüdükçe akış $\sqrt N$ ile büyür ve son katmanlarda her dalın göreli katkısı küçülür. GPT-2, akışa yazan katmanların (dikkatin $W_O$'su ve MLP'nin $W_2$'si) ağırlıklarını $1/\sqrt N$ ile ölçekler: dal varyansı $s^2/N$, toplam katkı $s^2$ — derinlikten bağımsız. nanoGPT'de `std = 0.02 / math.sqrt(2 * n_layer)`. (Varsayım: dallar ilişkisiz; eğitim ilerledikçe bu bozulur, ölçekleme yalnız başlangıç koşulunu ayarlar.) Deney l3'te "1/√L" seçeneği bunu gösteriyor.
:::

### Üretim: bir sonraki token nasıl seçilir? {#ornekleme}

Model her adımda bir dağılım verir; metni büyütmek için ondan **bir** token seçmek gerekir. Seçim kuralı (decoding) üretilen metnin niteliğini model kadar etkiler.

- **Açgözlü (greedy):** $\arg\max$. Deterministik; ama tekrara ve sıkıcı, döngüye giren metne eğilimlidir. En olası tek adım, en olası **dizi** değildir (ışın araması bunu kısmen düzeltir, M4.13).
- **Saf örnekleme:** $x\sim\mathrm{softmax}(\mathbf z)$. Çeşitli ama uzun metinde kuyruktaki düşük olasılıklı, anlamsız token'lar birikir. Holtzman ve ark. (2020) bunu "metin dejenerasyonu" olarak inceledi.
- **Sıcaklık:** $p_i\propto\exp(z_i/\tau)$.
- **Top-k:** en olası $k$ token dışındakileri at, kalanları yeniden normalize et (Fan ve ark. 2018).
- **Top-p (nucleus):** olasılığa göre sıralanmış token'lardan birikimli olasılığı $p$'ye ulaşan **en küçük** kümeyi tut (Holtzman ve ark. 2020). Model eminken küme küçük, kararsızken büyük olur — top-k'nın sabit $k$'sından daha uyarlanabilir.

:::türetim Sıcaklık olasılık oranlarını nasıl değiştirir?
İki token'ın olasılık oranı, normalizasyon sabiti sadeleşince:
$$\frac{p_i}{p_j}=\frac{e^{z_i/\tau}}{e^{z_j/\tau}}=\exp\Big(\frac{z_i-z_j}{\tau}\Big)=\Big(\frac{p_i^{(1)}}{p_j^{(1)}}\Big)^{1/\tau}.$$
Dolayısıyla $p_i^{(\tau)}\propto\big(p_i^{(1)}\big)^{1/\tau}$: $\tau=0.5$'te olasılıklar karesiyle orantılı hâle gelir. **Sıralama hiçbir $\tau>0$'da değişmez.** Limitler: $\tau\to0^+$'da en büyük logit'in oranı $\to\infty$, dağılım argmax'a (eşitlik yoksa) yakınsar; $\tau\to\infty$'da tüm oranlar $\to1$, dağılım düzgüne yakınsar.

**Sayısal örnek** (el hesabı alanındaki): logit'ler $[2,1,0]$. $\tau=1$: $p_1=e^2/(e^2+e+1)=7.389/11.107=0.665$. Top-k = 2: üçüncü atılır, $p_1=e^2/(e^2+e)=1/(1+e^{-1})=0.731$. $\tau=0.5$: logit'ler $[4,2,0]$, $p_1=54.6/(54.6+7.389+1)=0.867$.
:::

::lab l2

:::uyarı Sıcaklık bir "yaratıcılık" düğmesi değildir
Sıcaklığı yükseltmek dağılımın kuyruğunu kalınlaştırır; daha çeşitli ama daha sık **yanlış** ve tutarsız metin üretir. Kod, matematik ve olgusal sorularda düşük sıcaklık (ya da açgözlü) genellikle daha iyidir. Ayrıca $\tau$, top-k ve top-p'nin **uygulanma sırası** sonucu değiştirir (deneyde ve Hugging Face'te: sıcaklık → top-k → top-p).
:::

### KV önbelleği {#kv}

Çıkarımda model, ders slaytının vurguladığı gibi, bir RNN gibi durum taşımaz; **her turda yeniden çağrılır**. Saf uygulamada $t$. token'ı üretmek için $t-1$ uzunluğundaki öneki baştan işleriz. Ama nedensel maske yüzünden bu iş büyük ölçüde gereksizdir.

:::kanıt Eski konumların K, V ve çıktıları yeni token'la değişmez
Sızıntı kanıtındaki (yukarıda) iddianın aynısı: her katmanda $j$. konumun gizli durumu yalnız $x_{1:j}$'ye bağlıdır. Yeni bir token $x_{t}$ eklemek $j<t$ için $x_{1:j}$'yi değiştirmez; dolayısıyla $j<t$ konumlarının her katmandaki gizli durumları, dolayısıyla $\mathbf k_j=\mathrm{LN}(\mathbf x_j)W_K$ ve $\mathbf v_j=\mathrm{LN}(\mathbf x_j)W_V$, **aynen** aynıdır. ∎ (Koşul: model deterministik olmalı — eğitimdeki dropout kapalı; sayısal olarak da çoğu çekirdekte aynı işlem sırası korunmalı, aksi hâlde son basamaklarda fark olabilir.)
:::

Dolayısıyla her katmanda geçmiş K ve V'yi saklarsak, yeni adımda yalnız **yeni token'ın** $\mathbf q,\mathbf k,\mathbf v$'sini hesaplar, $\mathbf k,\mathbf v$'yi önbelleğe ekler ve yeni sorguyu önbelleğin tamamına karşı dikkatten geçiririz. Eski sorgular saklanmaz: yalnız eski token'ların çıktıları için gerekiyorlardı ve o çıktılar değişmez.

| | önbelleksiz | KV önbellekli |
|---|---|---|
| $t$. adımın işi | $t$ token'ın tamamı: $\approx 2Nt+2Ldt^2$ | tek token: $\approx 2N+4Ldt$ |
| $n$ token üretmek | $\approx N n^2+\tfrac23Ldn^3$ (karesel/kübik) | $\approx 2Nn+2Ldn^2$ |
| ek bellek | yok | $2\cdot L\cdot T\cdot d\cdot$bayt (K ve V) |

**Önbellek belleği.** Katman × token × 2 (K, V) × $d$ × bayt × batch. GPT-2 small, 1024 token, fp16: $2\cdot12\cdot1024\cdot768\cdot2$ B $=36$ MiB (pratik kartı P2). 7B ölçeğinde ($L=32$, $d=4096$) token başına 0.5 MiB; 32 eşzamanlı istek × 3072 token'da ≈ 48 GiB — ağırlıkların (≈ 13 GB) birkaç katı. Uzun bağlamlı çıkarımın darboğazı budur.

**İki evre.** İstemi işlemek (**prefill**) eğitimdeki gibi paraleldir ve hesapla sınırlıdır. Token token üretim (**decode**) her adımda tüm ağırlıkları ve tüm önbelleği bellekten okur ama az hesap yapar: **bellek bant genişliğiyle** sınırlıdır. Batch'i büyütmek decode'da ağırlık okumasını isteklere böler (verim artar), ama önbellek batch'le doğrusal büyür.

**K/V başlarını paylaşmak.** Çoklu sorgulu dikkat (**MQA**, Shazeer 2019) tüm sorgu başlarının **tek** bir K/V başını paylaşmasıdır; grup sorgulu dikkat (**GQA**, Ainslie ve ark. 2023) $h$ sorgu başını $g$ gruba böler, her grup bir K/V başı paylaşır. Önbellek $h/g$ kat küçülür; kalite kaybı GQA'da genelde küçüktür. LLaMA-2 70B ve sonrasının çoğu GQA kullanır.

::lab l4

### Eğitim tarifi {#egitim}

GPT-2/3 ve nanoGPT'nin kullandığı tarif (ayrıntılar M3.4, M3.9 ve M8):

| bileşen | tipik seçim | neden |
|---|---|---|
| optimizer | AdamW, $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$ | $\beta_2=0.95$ (0.999 yerine) seyrek büyük gradyanlara daha hızlı uyum sağlar |
| ağırlık çürümesi | 0.1, **yalnız 2-B tensörlere** (matrisler ve gömmeler; bias ve LN parametreleri hariç — nanoGPT'nin ayrımı) | ölçek parametrelerini sıfıra çekmek anlamsız |
| öğrenme oranı | doğrusal ısınma (ör. 2000 adım/375M token) + kosinüs azalma (tepe değerin %10'una) | Adam'ın ilk adımlardaki ikinci moment kestirimi gürültülü |
| gradyan kırpma | global norm 1.0 | ani sıçramalar (loss spike) |
| sayı biçimi | bf16 karışık duyarlık | bellek ve hız; bf16'nın üs aralığı fp32'yle aynı |
| batch | GPT-3 small: 0.5M token; GPT-3: 3.2M token | gradyan biriktirme ile sanal olarak büyütülür |
| dropout | GPT-1: 0.1; büyük modern ön-eğitimlerde çoğu kez 0 | tek epoch'ta aşırı uyum az |

:::uyarı Pratik kontroller
- İlk kayıp $\approx\ln V$ değilse başlatma yanlış.
- Tek bir küçük batch'i ezberletebiliyor musun (kayıp ≈ 0)? Ezberleyemiyorsan model ya da kayıp hesabında hata var (M3.9).
- Kayıp beklenmedik hızla düşüyorsa (özellikle ilk yüz adımda $\ll$ 2) **sızıntı** şüphelisi: maske ya da hedef kaydırması yanlış (P5).
- Doğrulama kaybını hep aynı token'larda ölç; perpleksiteyi farklı tokenleştiricili modeller arasında karşılaştırma (token başına değil, bayt/karakter başına ölç).
:::

### PyTorch'ta {#pytorch}

```python
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.h = h

    def forward(self, x):                                   # x: (B, T, d)
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (t.view(B, T, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))  # (B, h, T, d/h)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))
```

| amaç | PyTorch |
|---|---|
| token ve konum gömmesi | `nn.Embedding(V, d)`, `nn.Embedding(T, d)`; konumlar `torch.arange(T)` |
| nedensel dikkat (FlashAttention çekirdekleriyle) | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` |
| hazır katman | `nn.TransformerEncoderLayer(d, h, 4*d, activation="gelu", norm_first=True, batch_first=True)` + `mask=nn.Transformer.generate_square_subsequent_mask(T)` |
| ağırlık bağlama | `self.head.weight = self.wte.weight` |
| kayıp | `F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-1)` |
| örnekleme | `probs = F.softmax(logits[:, -1] / tau, -1)`; `torch.topk`; `torch.multinomial(probs, 1)` |
| top-p | `sorted, idx = probs.sort(descending=True)`; `cum = sorted.cumsum(-1)`; `sorted[cum - sorted > p] = 0` |
| hızlandırma | `torch.compile(model)`, `torch.autocast("cuda", dtype=torch.bfloat16)` |
| ağırlık çürümesi grupları | `p.dim() >= 2` olanlar bir grupta, diğerleri `weight_decay=0` |

:::yanılgı "`nn.TransformerDecoderLayer` GPT bloğudur."
Değildir: PyTorch'un `TransformerDecoderLayer`'ı orijinal kod çözücüdür, içinde kodlayıcıya bakan **çapraz dikkat** (`memory` argümanı) vardır. GPT bloğu, nedensel maskeli bir `TransformerEncoderLayer`'a (`norm_first=True`) karşılık gelir; "kodlayıcı" adı yanıltıcıdır, maske onu kod çözücü yapar.
:::

### Özet {#ozet}

- GPT = token + konum gömmesi → $L$ × (Pre-LN, nedensel çok başlı dikkat, artık; Pre-LN, MLP $d\to4d\to d$, artık) → son LN → bağlı çıkış katmanı. Orijinal Transformer'ın kod çözücüsünden çapraz dikkat çıkarılmış hâlidir.
- Eğitim hedefi zincir kuralından gelir: her konum bir sonraki token'ı tahmin eder; nedensel maske sayesinde $T$ tahmin tek ileri geçişte, sızıntısız eğitilir. İlk kayıp $\approx\ln V$.
- Blok başına $12d^2+13d$ parametre (MLP $\approx\tfrac23$); GPT-2 small $124\,439\,808$. Token başına ileri $\approx2N$, eğitim $\approx6N$ FLOP; dikkat terimi $T/(12d)$ oranında ekler.
- Pre-LN artık yolu temiz bırakır, ısınmaya daha az bağımlıdır; akış $\sqrt{L}$ ile büyür, $1/\sqrt{2L}$ artık başlatması ve $\mathrm{LN}_f$ bunu dengeler. LayerNorm'un öğrenilen parametreleri $d$ boyutlu $\boldsymbol\gamma,\boldsymbol\beta$'dır.
- Üretim: sıcaklık oranları $1/\tau$ kuvvetine yükseltir, sıralamayı değiştirmez; top-k/top-p kuyruğu keser.
- KV önbelleği eski K/V'yi saklayarak karesel yeniden hesaplamayı kaldırır; bedeli $2LTd$ sayılık bellek. MQA/GQA bunu küçültür.
