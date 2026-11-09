### Sabit gömmelerden bağlamsal gömmelere {#baglam}

M4.12'deki word2vec, GloVe ve fastText gibi öğrenilmiş **sabit** gömmelerin sorunu, bir kelimenin (ya da alt-kelime tokenının) anlamının **bağlama** bağlı olmasıdır. Çok anlamlı kelimeler bunu açıkça gösterir. "yüz" bir sayı ("yüz yıl"), bir organ ("yüzünü yıka") ya da bir eylemdir ("denizde yüz"). "çay" bir içecek ya da bir akarsudur. Sabit bir gömme bu anlamların sıklıklarıyla ağırlıklı **tek** bir ortalamasını verir ve hiçbirini iyi temsil etmez. **Bağlamsal gömme** fikri, ChatGPT dahil modern derin dil modellerinin temelidir. Bir kelimenin vektörü, içinde geçtiği cümlenin tamamına bakan bir ağın çıktısıdır.

Bu modül bağlamsal gömme çıkarıcılarını kronolojik sırayla ele alıyor: ELMo, GPT-1, BERT, BART, GPT-2/3, InstructGPT ve ChatGPT. Hepsinin ortak tarifi **ön eğitim + aktarım**tır. Etiketsiz büyük bir metinde kendi kendine üretilen bir görevle (bir sonraki kelime, maskelenmiş kelime, bozulmuş metin) bir ağ eğitilir. Ağın içinde oluşan temsiller sonra az etiketli görevlere aktarılır. Slaytın hatırlatması önemli: "Attention Is All You Need" (Transformer) NeurIPS **2017**'de yayımlandı. Aşağıdaki modellerin yıllarına bakınca, Transformer'ın bir yıl içinde alanı nasıl dönüştürdüğü görülür. ELMo hâlâ LSTM'dir, ondan birkaç ay sonra gelen GPT-1 ve BERT Transformer'dır.

### ELMo: çift yönlü LSTM dil modeli {#elmo}

**ELMo** (Embeddings from Language Models; Peters ve ark., NAACL 2018) büyük bir **çift yönlü dil modelinin** (biLM) iç durumlarını bağlamsal gömme olarak kullanır.

:::tanım biLM amacı
$N$ tokenlık bir dizi $(t_1,\dots,t_N)$ için ileri dil modeli $p(t_k\mid t_1,\dots,t_{k-1})$'i, geri dil modeli $p(t_k\mid t_{k+1},\dots,t_N)$'yi modeller. İkisi birlikte en çok olabilirlikle eğitilir:
$$\sum_{k=1}^N\Big[\log p(t_k\mid t_{<k};\Theta_x,\overrightarrow\Theta_{\text{LSTM}},\Theta_s)+\log p(t_k\mid t_{>k};\Theta_x,\overleftarrow\Theta_{\text{LSTM}},\Theta_s)\Big].$$
Token temsili $\Theta_x$ ve softmax katmanı $\Theta_s$ iki yönde paylaşılır; LSTM'ler ayrıdır.
:::

İleri LSTM baştan başlayarak her kelimeden bir sonrakini, geri LSTM sondan başlayarak her kelimeden bir öncekini tahmin eder. Mimari ayrıntıları:
- iki katmanlı biLSTM, katman başına 4096 birim ve 512 boyutlu izdüşüm;
- birinci katmandan ikinciye bir **artık bağlantı**;
- eğitim verisi 1B Word Benchmark (≈ 800M token), 10 epoch.

**Bağlamsal gömme**, tüm katmanların gizli durumlarının görev başına öğrenilen bir birleşimidir. $k$. token için $\mathbf h_{k,0}$ bağlamdan bağımsız token temsili, $\mathbf h_{k,j}=[\overrightarrow{\mathbf h}_{k,j};\overleftarrow{\mathbf h}_{k,j}]$ ise $j$. LSTM katmanının iki yönünün birleşimi olsun:
$$\mathrm{ELMo}_k^{\text{görev}}=\gamma^{\text{görev}}\sum_{j=0}^{L}s_j^{\text{görev}}\,\mathbf h_{k,j},\qquad s^{\text{görev}}=\mathrm{softmax}(\text{öğrenilen ağırlıklar}) .$$
$\gamma$ tüm vektörü ölçekler. Katmanlar farklı bilgi taşıdığı için karışım önemlidir: yazarlar alt katmanın sözdizimini (sözcük türü etiketleme), üst katmanın anlamı (kelime anlamı ayrımı) daha iyi kodladığını buldu.

<figure class="fig wide"><svg viewBox="0 0 640 332" role="img" aria-label="ELMo: her kelime karakter CNN'iyle temsil edilir, iki katmanlı ileri ve geri LSTM'lerden geçer; bağlamsal gömme katmanların ağırlıklı toplamıdır"><defs><marker id="e1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="e2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker><marker id="e3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><text x="130" y="282" text-anchor="middle">yüz</text><rect x="90" y="220" width="80" height="26" rx="5" class="soft-p"/><text x="130.0" y="237.0" text-anchor="middle" class="sm">karakter CNN</text><line x1="130" y1="266" x2="130" y2="248" class="ink" marker-end="url(#e1)"/><rect x="90" y="172" width="80" height="26" rx="5" class="soft-a"/><text x="130.0" y="189.0" text-anchor="middle" class="sm">LSTM₁ →</text><rect x="90" y="140" width="80" height="26" rx="5" class="soft-b"/><text x="130.0" y="157.0" text-anchor="middle" class="sm">LSTM₁ ←</text><rect x="90" y="92" width="80" height="26" rx="5" class="soft-a"/><text x="130.0" y="109.0" text-anchor="middle" class="sm">LSTM₂ →</text><rect x="90" y="60" width="80" height="26" rx="5" class="soft-b"/><text x="130.0" y="77.0" text-anchor="middle" class="sm">LSTM₂ ←</text><line x1="130" y1="220" x2="130" y2="200" class="ink" marker-end="url(#e1)"/><line x1="130" y1="140" x2="130" y2="120" class="ink" marker-end="url(#e1)"/><line x1="172" y1="185" x2="258" y2="185" class="ca" marker-end="url(#e2)"/><line x1="172" y1="105" x2="258" y2="105" class="ca" marker-end="url(#e2)"/><text x="130" y="40" text-anchor="middle" class="sm">h₀, h₁, h₂</text><text x="300" y="282" text-anchor="middle">yıl</text><rect x="260" y="220" width="80" height="26" rx="5" class="soft-p"/><text x="300.0" y="237.0" text-anchor="middle" class="sm">karakter CNN</text><line x1="300" y1="266" x2="300" y2="248" class="ink" marker-end="url(#e1)"/><rect x="260" y="172" width="80" height="26" rx="5" class="soft-a"/><text x="300.0" y="189.0" text-anchor="middle" class="sm">LSTM₁ →</text><rect x="260" y="140" width="80" height="26" rx="5" class="soft-b"/><text x="300.0" y="157.0" text-anchor="middle" class="sm">LSTM₁ ←</text><rect x="260" y="92" width="80" height="26" rx="5" class="soft-a"/><text x="300.0" y="109.0" text-anchor="middle" class="sm">LSTM₂ →</text><rect x="260" y="60" width="80" height="26" rx="5" class="soft-b"/><text x="300.0" y="77.0" text-anchor="middle" class="sm">LSTM₂ ←</text><line x1="300" y1="220" x2="300" y2="200" class="ink" marker-end="url(#e1)"/><line x1="300" y1="140" x2="300" y2="120" class="ink" marker-end="url(#e1)"/><line x1="342" y1="185" x2="428" y2="185" class="ca" marker-end="url(#e2)"/><line x1="342" y1="105" x2="428" y2="105" class="ca" marker-end="url(#e2)"/><line x1="258" y1="153" x2="172" y2="153" class="cb" marker-end="url(#e3)"/><line x1="258" y1="73" x2="172" y2="73" class="cb" marker-end="url(#e3)"/><text x="300" y="40" text-anchor="middle" class="sm">h₀, h₁, h₂</text><text x="470" y="282" text-anchor="middle">önce</text><rect x="430" y="220" width="80" height="26" rx="5" class="soft-p"/><text x="470.0" y="237.0" text-anchor="middle" class="sm">karakter CNN</text><line x1="470" y1="266" x2="470" y2="248" class="ink" marker-end="url(#e1)"/><rect x="430" y="172" width="80" height="26" rx="5" class="soft-a"/><text x="470.0" y="189.0" text-anchor="middle" class="sm">LSTM₁ →</text><rect x="430" y="140" width="80" height="26" rx="5" class="soft-b"/><text x="470.0" y="157.0" text-anchor="middle" class="sm">LSTM₁ ←</text><rect x="430" y="92" width="80" height="26" rx="5" class="soft-a"/><text x="470.0" y="109.0" text-anchor="middle" class="sm">LSTM₂ →</text><rect x="430" y="60" width="80" height="26" rx="5" class="soft-b"/><text x="470.0" y="77.0" text-anchor="middle" class="sm">LSTM₂ ←</text><line x1="470" y1="220" x2="470" y2="200" class="ink" marker-end="url(#e1)"/><line x1="470" y1="140" x2="470" y2="120" class="ink" marker-end="url(#e1)"/><line x1="428" y1="153" x2="342" y2="153" class="cb" marker-end="url(#e3)"/><line x1="428" y1="73" x2="342" y2="73" class="cb" marker-end="url(#e3)"/><text x="470" y="40" text-anchor="middle" class="sm">h₀, h₁, h₂</text><text x="20" y="20" class="sm">ELMo<tspan baseline-shift="-3" font-size="75%">k</tspan> = γ · Σ<tspan baseline-shift="-3" font-size="75%">j</tspan> softmax(s)<tspan baseline-shift="-3" font-size="75%">j</tspan> · h<tspan baseline-shift="-3" font-size="75%">k,j</tspan>  (j = 0: karakter temsili, 1–2: LSTM katmanları, iki yön uç uca)</text><text x="20" y="306" class="sm">İleri LSTM sonraki kelimeyi, geri LSTM önceki kelimeyi tahmin eder;</text><text x="20" y="322" class="sm">iki yön ayrı eğitilir, temsiller sonradan birleştirilir.</text></svg><figcaption>Şekil 1 — ELMo'nun yapısı (Peters ve ark. 2018). Girdi, kelimenin karakterlerinden bir CNN ile hesaplanan bağlamdan bağımsız temsildir; üstünde iki katmanlı, ileri (mavi oklar) ve geri (turuncu oklar) LSTM'ler vardır. Bir görev, her kelime için üç katmanın temsillerini öğrenilmiş ağırlıklarla karıştırır. 'yüz' kelimesinin vektörü artık 'yıl' ve 'önce'ye bağlıdır.</figcaption></figure>

:::uyarı Kaynak düzeltmesi: ELMo'nun girdisi
Slaytlar ELMo'nun girdilerini "normal embedding'ler (one-hot, word2vec)" olarak gösteriyor. ELMo'nun girdisi önceden eğitilmiş kelime vektörleri değildir. Her kelimenin **karakterlerinden** hesaplanan, bağlamdan bağımsız bir temsildir: 2048 karakter n-gram konvolüsyon filtresi, iki highway katmanı ve 512 boyuta doğrusal izdüşüm. Bu sayede ELMo sözlük dışı kelimelere de temsil üretir (M4.12'deki fastText fikrine benzer). Görev modelinde ise ELMo vektörü çoğu zaman görevin kendi kelime embedding'leriyle (ör. GloVe) **uç uca eklenir**; slaytlardaki "normal embedding'lerin yerine geçer" ifadesi bu bakımdan da kesin değildir. Ayrıca ELMo yalnız Washington Üniversitesi'nin değil, Allen Yapay Zekâ Enstitüsü (AI2) ile Washington Üniversitesi'nin ortak çalışmasıdır.
:::

ELMo **kendi başına** bir aşağı akış görevi yapmaz. Ama düzenli embedding'lerin yerine (ya da yanına) konarak görevlerin başarımını artırır. Örneğin duygu analizi modeline düz embedding'leri doğrudan vermek yerine, onları ELMo'dan geçirip ELMo'nun çıktı gömmelerini görev modeline verirsin. biLM ağırlıkları dondurulur, yalnız $s$, $\gamma$ ve görev modeli eğitilir. ELMo altı farklı görevde (soru cevaplama, metin içerimi, anlamsal rol etiketleme, eş-gönderim, adlandırılmış varlık, duygu) o günün en iyi sonuçlarını geçti.

:::sezgi "Sığ" çift yönlülük
ELMo'nun iki yönü **ayrı** eğitilir ve yalnız en sonda uç uca eklenir. İleri LSTM sağ bağlamı hiç görmez, geri LSTM sol bağlamı hiç görmez. Hiçbir katmanda bir kelimenin temsili iki yandaki bağlamı **birlikte** işlemez. BERT'in "derin çift yönlü" iddiası tam olarak bu sınıra karşılık gelir.
:::

### GPT-1: yalnız kod çözücülü ön eğitim {#gpt1}

**GPT-1** (Generative Pre-Training; Radford ve ark., OpenAI, Haziran 2018), M4.14'teki Transformer'ın **kod çözücüsünü** kullanır, ama kodlayıcı yoktur. Dolayısıyla kodlayıcı–kod çözücü (çapraz) dikkati de yoktur: her blok maskeli öz-dikkat + FFN'dir. Model yalnız **ileri** dil modellemesiyle, yani bir sonraki tokenı tahmin ederek eğitilir. M4.7'de kurduğun GPT'nin atasıdır.

- **Veri:** BooksCorpus (Toronto Kitap Derlemi), çeşitli türlerde 7 000'den fazla yayımlanmamış kitap. Kitaplar Smashwords adlı bağımsız e-kitap dağıtım sitesinden toplanmıştır. Uzun ve bitişik metin içerdiği için model uzun menzilli bağlamı öğrenir.
- **Model:** 12 katman, 768 gizli boyut, 12 dikkat başı, FFN 3072, bağlam 512 token, 40 000 birleştirmeli BPE (M4.5), öğrenilen konum embedding'leri, GELU.
- **Eğitim:** 64'lük batch'lerle 100 epoch.

:::uyarı Kaynak düzeltmesi: GPT-1'in parametre sayısı
Slayt GPT-1 için "110 M parametre" diyor. GPT-1 yaygın olarak **117M** parametreyle bilinir; bu dersin ch_16 zaman çizelgesi de 117M yazar. 110M, **BERT-base**'in sayısıdır. BERT-base, karşılaştırma için bilerek GPT-1 ile aynı boyutta (12/768/12) seçilmişti; iki sayı arasındaki fark esas olarak sözlük boylarından gelir (GPT-1'de ≈ 40k BPE, BERT'te 30k WordPiece).
:::

**İnce ayar.** Göreve özel bir mimari yoktur. Son katmanın son tokendaki durumu $\mathbf h_l^m$ üzerine **tek bir doğrusal katman** eklenir: $P(y\mid x)=\mathrm{softmax}(\mathbf h_l^m\mathbf W_y)$. Ağın **tüm** ağırlıkları (12 katmanın hepsi) denetimli etiketlerle güncellenir. Yazarlar ince ayara yardımcı bir dil modelleme kaybı ekledi: $L_3=L_2+\lambda L_1$, $\lambda=0.5$. Bu kayıp genellemeyi iyileştirdi ve yakınsamayı hızlandırdı.

**Girdi dönüşümleri.** Aşağı akış girdilerinin hepsi token dizilerine çevrilir:
- **İçerim:** öncül ve hipotez bir ayraç tokenıyla birleştirilir: `[başla] öncül $ hipotez [çıkar]`.
- **Benzerlik:** iki sıra da işlenir, temsiller toplanır.
- **Çoktan seçmeli:** her seçenek ayrı bir dizi olur, skorlar softmax'tan geçer.

Böylece aynı önceden eğitilmiş ağ, mimari değişmeden farklı görevlere uyar. GPT-1 incelenen 12 görevin 9'unda o günün en iyi sonuçlarını geçti.

:::uyarı Kaynak notu: "GPT-1 üretken bir model olarak tasarlanmadı"
Slayt GPT-1'in üretken bir model olarak değil bağlamsal gömme elde etmek için tasarlandığını söylüyor. Doğru olan yarısı şudur: makalenin amacı, üretken bir **ön eğitimle** öğrenilen temsilleri **ayırt edici** görevlere aktarmaktı ("Improving Language Understanding by Generative Pre-Training"). Ama ön eğitim amacının kendisi üretkendir: GPT-1 bir dil modelidir ve metin üretebilir. Üretim yeteneğinin bir kullanım biçimi olarak öne çıkması GPT-2 ile gelir.
:::

### BERT: derin çift yönlü kodlayıcı {#bert}

**BERT** (Bidirectional Encoder Representations from Transformers; Devlin ve ark., Google). arXiv'de Ekim 2018'de yayımlandı, NAACL 2019'da sunuldu; slayttaki "2019" konferans yılıdır. Fikri iki öncekinin birleşimidir: ELMo gibi **çift yönlü**, GPT gibi **Transformer**. BERT, Transformer'ın yalnız **kodlayıcısını** kullanır. Kodlayıcıda nedensel maske yoktur, her token her tokena bakar. BERT hem ELMo'dan hem GPT-1'den daha iyiydi.

:::türetim Neden "sonraki token" amacı çift yönlü kodlayıcıyla kullanılamaz?
Çok katmanlı bir kodlayıcıda $\ell$. katmanda $i$. konum tüm konumlara bakar; buna $i$ konumu da dahildir. Girdideki $t_i$ tokenının bilgisi birinci katmanda tüm konumlara dağılır, ikinci katmanda $i$. konum bu bilgiyi diğer konumlardan geri alır. Amaç $t_i$'yi $i$. konumun (ya da $i-1$.'in) son temsilinden tahmin etmek olsaydı model cevabı girdiden **kopyalayabilirdi**: kayıp sıfıra iner, hiçbir şey öğrenilmez. ELMo bunu iki yönü ayrı eğiterek önler; GPT nedensel maskeyle sağ bağlamı keserek. BERT ise tahmin edilecek tokenı girdiden **siler**: "maskeli dil modeli".
:::

BERT iki görevle ön eğitilir:
1. **Maskeli token tahmini** (masked language model, MLM). Tokenların %15'i rastgele seçilir; model bunları komşu kelimelerden tahmin eder. Seçilen tokenların %80'i `[MASK]` ile değiştirilir, %10'u rastgele bir tokenla değiştirilir, %10'u aynı kalır. Kayıp yalnız seçilen konumlarda hesaplanır.
2. **Sonraki cümle tahmini** (next sentence prediction, NSP). İki cümle (A, B) verilir. %50 olasılıkla B gerçekten A'dan sonra gelen cümledir, %50 olasılıkla derlemden rastgele bir cümledir. Model, `[CLS]` tokenının son durumundan bu ikisini ayırt eder.

:::örnek El hesabı: MLM
512 tokenlık bir dizide seçilen token sayısının beklentisi $512\cdot0.15=\mathbf{76.8}$. Bunların %80'i (61.4) `[MASK]` olur, %10'u (**7.68**) rastgele bir tokenla değişir, %10'u (7.68) aynı kalır. **Neden 80/10/10?** İnce ayarda girdide `[MASK]` hiç görülmez. Yalnız `[MASK]` kullanılsaydı model "tahmin etmem gereken yer `[MASK]`'ın olduğu yerdir" diye öğrenir ve diğer tokenların temsilini önemsemezdi. Rastgele ve değişmeyen tokenlar modeli **her** tokenın temsilini bağlama göre doğru tutmaya zorlar: hangi tokenın bozulduğunu bilemez. **Bedeli:** kayıp dizinin yalnız %15'inden sinyal alır, nedensel LM her konumdan alır. Bu, ELECTRA'nın (ek x3) çıkış noktasıdır.
:::

**Girdi.** Her tokenın vektörü üç embedding'in toplamıdır: token (30 000 WordPiece sözlüğü), segment (A ya da B cümlesi) ve öğrenilen konum (512). Dizi `[CLS]` ile başlar, cümleler `[SEP]` ile ayrılır.

**Boyutlar ve veri.**

| model | katman | gizli | baş | parametre |
|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

Ön eğitim verisi BooksCorpus (800M kelime) ve İngilizce Vikipedi'dir (2 500M kelime).

:::türetim BERT-base'in parametre sayımı
$V=30\,522$, 512 konum, 2 segment, $d=768$, 12 katman, FFN 3072:
- embedding'ler + embedding LN: $(30\,522+512+2)\cdot768+2\cdot768=23\,837\,184$;
- katman: M4.15'teki ViT bloğunun aynısı (MSA + FFN + 2 LN; yalnız LN'nin yeri farklı) $=7\,087\,872$, × 12 = $85\,054\,464$;
- pooler ([CLS] üzerinde $d\times d$ + tanh): $768^2+768=590\,592$.

Toplam $\mathbf{109\,482\,240}\approx110$M. MLM başı, çıktı ağırlığını token embedding'iyle paylaştığı için yalnız küçük bir dönüşüm ve bias ekler.
:::

**İnce ayar.** BERT aşağı akış görevleri için ince ayarlanmalıdır. Sınıflandırmada `[CLS]` tokenının son durumu üzerine bir doğrusal katman konur. Soru cevaplamada her tokenın "cevabın başı/sonu" olma skoru hesaplanır. BERT-large GLUE kıyaslamasında 80.5 puana ulaştı (önceki en iyiden 7.7 puan fazla); SQuAD v1.1'de test F1 93.2 aldı. Sonraki çalışmalar tarifi iyileştirdi: RoBERTa (Liu ve ark. 2019) NSP'nin gereksiz olduğunu, daha uzun eğitimin ve daha çok verinin asıl kazancı getirdiğini gösterdi.

### BART: gürültü giderici kodlayıcı–kod çözücü {#bart}

**BART** (Bidirectional and Auto-Regressive Transformers; Lewis ve ark., Facebook AI; arXiv Ekim 2019, ACL 2020). BART, bir dizi→dizi modeliyle kurulmuş bir **gürültü giderici otomatik kodlayıcıdır** (denoising autoencoder): metin bir gürültü fonksiyonuyla bozulur, model özgün metni yeniden üretmeyi öğrenir. Mimari, M4.14'teki tam kodlayıcı–kod çözücü Transformer'dır. Kodlayıcı bozulmuş metni çift yönlü okur (BERT gibi), kod çözücü özgün metni soldan sağa üretir (GPT gibi). Bu yüzden çok çeşitli görevlere uygulanabilir: hem anlama (sınıflandırma) hem üretim (özetleme, çeviri).

<figure class="fig wide"><svg viewBox="0 0 640 260" role="img" aria-label="Üç ön eğitim ailesinin dikkat desenleri: BERT'te tam kare, GPT'de alt üçgen, BART'ta tam kodlayıcı ile nedensel kod çözücü ve çapraz dikkat"><text x="30" y="30" class="ta">BERT: yalnız kodlayıcı</text><text x="30" y="46" class="sm">her token her tokena bakar</text><rect x="30" y="60" width="18" height="18" class="ca-f"/><rect x="50" y="60" width="18" height="18" class="ca-f"/><rect x="70" y="60" width="18" height="18" class="ca-f"/><rect x="90" y="60" width="18" height="18" class="ca-f"/><rect x="110" y="60" width="18" height="18" class="ca-f"/><rect x="130" y="60" width="18" height="18" class="ca-f"/><rect x="30" y="80" width="18" height="18" class="ca-f"/><rect x="50" y="80" width="18" height="18" class="ca-f"/><rect x="70" y="80" width="18" height="18" class="ca-f"/><rect x="90" y="80" width="18" height="18" class="ca-f"/><rect x="110" y="80" width="18" height="18" class="ca-f"/><rect x="130" y="80" width="18" height="18" class="ca-f"/><rect x="30" y="100" width="18" height="18" class="ca-f"/><rect x="50" y="100" width="18" height="18" class="ca-f"/><rect x="70" y="100" width="18" height="18" class="ca-f"/><rect x="90" y="100" width="18" height="18" class="ca-f"/><rect x="110" y="100" width="18" height="18" class="ca-f"/><rect x="130" y="100" width="18" height="18" class="ca-f"/><rect x="30" y="120" width="18" height="18" class="ca-f"/><rect x="50" y="120" width="18" height="18" class="ca-f"/><rect x="70" y="120" width="18" height="18" class="ca-f"/><rect x="90" y="120" width="18" height="18" class="ca-f"/><rect x="110" y="120" width="18" height="18" class="ca-f"/><rect x="130" y="120" width="18" height="18" class="ca-f"/><rect x="30" y="140" width="18" height="18" class="ca-f"/><rect x="50" y="140" width="18" height="18" class="ca-f"/><rect x="70" y="140" width="18" height="18" class="ca-f"/><rect x="90" y="140" width="18" height="18" class="ca-f"/><rect x="110" y="140" width="18" height="18" class="ca-f"/><rect x="130" y="140" width="18" height="18" class="ca-f"/><rect x="30" y="160" width="18" height="18" class="ca-f"/><rect x="50" y="160" width="18" height="18" class="ca-f"/><rect x="70" y="160" width="18" height="18" class="ca-f"/><rect x="90" y="160" width="18" height="18" class="ca-f"/><rect x="110" y="160" width="18" height="18" class="ca-f"/><rect x="130" y="160" width="18" height="18" class="ca-f"/><text x="240" y="30" class="ta">GPT: yalnız kod çözücü</text><text x="240" y="46" class="sm">nedensel: yalnız geçmiş</text><rect x="240" y="60" width="18" height="18" class="ca-f"/><rect x="260" y="60" width="18" height="18" class="mute-f"/><rect x="280" y="60" width="18" height="18" class="mute-f"/><rect x="300" y="60" width="18" height="18" class="mute-f"/><rect x="320" y="60" width="18" height="18" class="mute-f"/><rect x="340" y="60" width="18" height="18" class="mute-f"/><rect x="240" y="80" width="18" height="18" class="ca-f"/><rect x="260" y="80" width="18" height="18" class="ca-f"/><rect x="280" y="80" width="18" height="18" class="mute-f"/><rect x="300" y="80" width="18" height="18" class="mute-f"/><rect x="320" y="80" width="18" height="18" class="mute-f"/><rect x="340" y="80" width="18" height="18" class="mute-f"/><rect x="240" y="100" width="18" height="18" class="ca-f"/><rect x="260" y="100" width="18" height="18" class="ca-f"/><rect x="280" y="100" width="18" height="18" class="ca-f"/><rect x="300" y="100" width="18" height="18" class="mute-f"/><rect x="320" y="100" width="18" height="18" class="mute-f"/><rect x="340" y="100" width="18" height="18" class="mute-f"/><rect x="240" y="120" width="18" height="18" class="ca-f"/><rect x="260" y="120" width="18" height="18" class="ca-f"/><rect x="280" y="120" width="18" height="18" class="ca-f"/><rect x="300" y="120" width="18" height="18" class="ca-f"/><rect x="320" y="120" width="18" height="18" class="mute-f"/><rect x="340" y="120" width="18" height="18" class="mute-f"/><rect x="240" y="140" width="18" height="18" class="ca-f"/><rect x="260" y="140" width="18" height="18" class="ca-f"/><rect x="280" y="140" width="18" height="18" class="ca-f"/><rect x="300" y="140" width="18" height="18" class="ca-f"/><rect x="320" y="140" width="18" height="18" class="ca-f"/><rect x="340" y="140" width="18" height="18" class="mute-f"/><rect x="240" y="160" width="18" height="18" class="ca-f"/><rect x="260" y="160" width="18" height="18" class="ca-f"/><rect x="280" y="160" width="18" height="18" class="ca-f"/><rect x="300" y="160" width="18" height="18" class="ca-f"/><rect x="320" y="160" width="18" height="18" class="ca-f"/><rect x="340" y="160" width="18" height="18" class="ca-f"/><text x="450" y="30" class="ta">BART: kodlayıcı–kod çözücü</text><text x="450" y="46" class="sm">mavi: öz-dikkat, turuncu: çapraz</text><rect x="450" y="60" width="18" height="18" class="ca-f"/><rect x="470" y="60" width="18" height="18" class="ca-f"/><rect x="490" y="60" width="18" height="18" class="ca-f"/><rect x="450" y="80" width="18" height="18" class="ca-f"/><rect x="470" y="80" width="18" height="18" class="ca-f"/><rect x="490" y="80" width="18" height="18" class="ca-f"/><rect x="450" y="100" width="18" height="18" class="ca-f"/><rect x="470" y="100" width="18" height="18" class="ca-f"/><rect x="490" y="100" width="18" height="18" class="ca-f"/><rect x="520" y="130" width="18" height="18" class="ca-f"/><rect x="450" y="130" width="18" height="18" class="cb-f"/><rect x="540" y="130" width="18" height="18" class="mute-f"/><rect x="470" y="130" width="18" height="18" class="cb-f"/><rect x="560" y="130" width="18" height="18" class="mute-f"/><rect x="490" y="130" width="18" height="18" class="cb-f"/><rect x="520" y="150" width="18" height="18" class="ca-f"/><rect x="450" y="150" width="18" height="18" class="cb-f"/><rect x="540" y="150" width="18" height="18" class="ca-f"/><rect x="470" y="150" width="18" height="18" class="cb-f"/><rect x="560" y="150" width="18" height="18" class="mute-f"/><rect x="490" y="150" width="18" height="18" class="cb-f"/><rect x="520" y="170" width="18" height="18" class="ca-f"/><rect x="450" y="170" width="18" height="18" class="cb-f"/><rect x="540" y="170" width="18" height="18" class="ca-f"/><rect x="470" y="170" width="18" height="18" class="cb-f"/><rect x="560" y="170" width="18" height="18" class="ca-f"/><rect x="490" y="170" width="18" height="18" class="cb-f"/><text x="520" y="95" class="sm">kodlayıcı</text><text x="520" y="198" class="sm">kod çözücü</text><text x="30" y="230" class="sm">amaç: MLM + NSP</text><text x="240" y="230" class="sm">amaç: sonraki token (CLM)</text><text x="450" y="230" class="sm">amaç: bozulmuş metni düzelt</text><text x="30" y="248" class="sm">anlama, sınıflandırma</text><text x="240" y="248" class="sm">üretim, istemle kullanım</text><text x="450" y="248" class="sm">özetleme, çeviri</text></svg><figcaption>Şekil 2 — BERT, GPT ve BART aynı Transformer bloklarını farklı maskelerle ve farklı ön eğitim amaçlarıyla kullanır. Satır: sorgu, sütun: anahtar; renkli: izinli. Çift yönlü dikkat bir tokenın temsiline sağ bağlamı da katar, ama bu yüzden 'sonraki tokenı tahmin et' amacı kullanılamaz (token kendini görürdü): BERT maskelemeye başvurur.</figcaption></figure>

BART'ın denediği gürültü fonksiyonları:
- **token maskeleme** (BERT gibi);
- **token silme:** model hangi konumların eksik olduğunu da bulmalı;
- **metin doldurma** (text infilling): uzunluğu Poisson($\lambda=3$) dağılımlı aralıklar seçilir ve her aralık **tek** bir `<mask>` ile değiştirilir. 0 uzunluklu bir aralık, araya bir maske eklemek demektir. Model aralıkta kaç token eksik olduğunu da tahmin etmelidir.
- **cümle karıştırma:** belgedeki cümlelerin sırası karıştırılır.
- **belge döndürme:** belge rastgele bir tokendan başlayacak biçimde döndürülür; model belgenin gerçek başını bulmalıdır.

En iyi sonuçları metin doldurma ve cümle karıştırmanın birleşimi verdi. BART özellikle özetlemede o günün en iyi sonuçlarını geçti.

Aşağıdaki deneyde aynı Türkçe paragrafa BERT'in MLM'sini, BART'ın beş gürültüsünü ve ELECTRA'nın "değiştirilmiş token tespitini" uygula. Her amacın kaybının dizinin ne kadarından sinyal aldığını karşılaştır.

::lab l2

### GPT-2 ve GPT-3: ölçek ve istemle kullanım {#gpt23}

**GPT-2** (Radford ve ark., Şubat 2019). GPT-1'den yaklaşık 10 kat büyüktür (1.5B parametre, tam oranı ≈ 13): 48 katman, 1600 gizli boyut, bağlam 1024 token, 50 257 tokenlık bayt düzeyinde BPE. LayerNorm her alt bloğun girişine taşınmıştır (Pre-LN; M4.7). Eğitim verisi **WebText**'tir: Reddit gönderilerinde en az 3 karma almış dış bağlantılardan toplanan ≈ 8 milyon belge, 40 GB metin. Karma eşiği, insanların "ilginç ya da yararlı" bulduğu sayfalar için ucuz bir kalite filtresidir. Model dört boyutta eğitildi (117M, 345M, 762M, 1.5B). En büyüğü, kötüye kullanım endişesiyle birkaç ay içinde kademeli olarak yayımlandı.

GPT-2'nin vurgusu **ince ayarsız**, yalnız **istemle** (prompt) çok görev yapmaktı: soru cevaplama, çeviri, özetleme. Görev doğal dilde ya da örnek biçimiyle anlatılır. Özetleme için metnin sonuna "TL;DR:" eklenir. Çeviri için birkaç "İngilizce cümle = Fransızca cümle" çifti verilir, sonra yeni bir İngilizce cümle ve "=" gelir. Sonuçlar karışıktı. GPT-2 sekiz dil modelleme veri kümesinin yedisinde sıfır örnekli (hiç ince ayar görmeden) en iyi sonucu verdi. Okuduğunu anlamada denetimli temel modellere yaklaştı. Özetleme ve çeviride ise denetimli sistemlerin çok gerisindeydi. Asıl heyecan verici olan, makalenin model boyutuna göre sıfır örnekli başarım grafiğindeydi: her görevde başarım model boyuyla **düzenli** artıyordu ve en büyük model hâlâ doymamıştı. Slayttaki tek boynuzlu atlar örneği de bu dönemin akılda kalan gösterimidir: model bir paragraflık hayalî bir haber girişinden tutarlı, uzun ve tamamen uydurma bir haber metni üretmişti. (Metin OpenAI'ın blog yazısındadır; burada aktarılmıyor.)

**GPT-3** (Brown ve ark., Mayıs 2020). GPT-2'den yaklaşık 100 kat büyüktür (175B parametre, oran ≈ 117): 96 katman, 12 288 gizli boyut, 96 baş, bağlam 2048 token. Eğitim verisinin %60'ı filtrelenmiş Common Crawl'dur (45 TB ham metinden 570 GB'a süzülmüş). Geri kalanı WebText2, iki kitap derlemi ve Vikipedi'den gelir; toplam 300 milyar token görülmüştür.

:::türetim Parametre sayısı ≈ $12Ld^2$
M4.7'deki sayımla bir blokta dikkat $4d^2$, MLP ($4d$ genişlik) $8d^2$ parametre taşır; biaslar ve LN $O(d)$'dir. $L$ blok: $\approx12Ld^2$. GPT-2: $12\cdot48\cdot1600^2=\mathbf{1.47\times10^9}$; embedding ($50\,257\cdot1600\approx80$M) eklenince 1.5B. GPT-3: $12\cdot96\cdot12\,288^2=\mathbf{1.74\times10^{11}}$, embedding'le ≈ 175B. Yaklaşım, $d$ sözlük boyuna göre büyüdükçe iyileşir.
:::

GPT-3 **sıfır örnekli** (yalnız görev tanımı), **tek örnekli** (bir örnek) ve **az örnekli** (bağlam penceresine sığan birkaç örnek, tipik 10–100) görevlerde ince ayarsız çok iyiydi. Bu, **bağlam içi öğrenmedir** (in-context learning): modelin ağırlıkları değişmez; görev, istemin içindeki örneklerden ileri geçiş sırasında "öğrenilir". Slaytın ifadesiyle bu kırılma noktasıydı: büyük bir dil modeli, her görev için ayrı ince ayar gerektirmeyen genel amaçlı bir araca dönüştü.

İstemle kullanımda model metni token token **üretir**. Her adımda logitlerden bir dağılım hesaplanır ve bir sonraki token bu dağılımdan seçilir. Seçim kuralı (açgözlü, sıcaklıklı örnekleme, top-k, top-p) M4.7'de ayrıntılı işlendi. Aşağıdaki deneyde sıcaklığın aynı logitlerden ne kadar farklı dağılımlar ürettiğini gör.

::lab l1

### InstructGPT ve RLHF {#rlhf}

GPT-3'ün sorunu, kullanıcının **niyetini** yakalayamamasıydı. Bir metin girdisi verilince akıcı ve insan yazısına çok benzeyen metin üretmeye devam ederdi, ama bu metin tamamen uydurma olabilirdi. "Yolu geçen tavuklar hakkında bir fıkra yaz" gibi basit bir isteği yerine getiremezdi. Genellikle girdideki kelimeleri tekrarlayan anlamsız bir devam üretirdi. Sebep şudur: model "internetteki metni sürdür" diye eğitilmişti, "talimatı yerine getir" diye değil. Bir sonraki adım **InstructGPT** oldu (Ocak 2022; Ouyang ve ark.). InstructGPT insan geri bildirimiyle pekiştirmeli öğrenme (reinforcement learning from human feedback, **RLHF**) ile eğitildi.

<figure class="fig wide"><svg viewBox="0 0 640 192" role="img" aria-label="RLHF'nin üç adımı: denetimli ince ayar, insan tercihlerinden ödül modeli, KL cezalı ödülle PPO; DPO ikinci ve üçüncü adımı tek bir kayıpla birleştirir"><defs><marker id="e4" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="20" y="40" width="190" height="80" rx="8" class="soft-a"/><text x="115" y="64" text-anchor="middle" class="sm">1. denetimli ince ayar (SFT)</text><text x="115" y="86" text-anchor="middle" class="sm">istem + insan yazımı yanıt</text><text x="115" y="104" text-anchor="middle" class="sm">→ CE</text><line x1="212" y1="80" x2="228" y2="80" class="ink" marker-end="url(#e1)"/><rect x="230" y="40" width="190" height="80" rx="8" class="soft-b"/><text x="325" y="64" text-anchor="middle" class="sm">2. ödül modeli (RM)</text><text x="325" y="86" text-anchor="middle" class="sm">yanıtları insanlar sıralar</text><text x="325" y="104" text-anchor="middle" class="sm">→ Bradley–Terry</text><line x1="422" y1="80" x2="438" y2="80" class="ink" marker-end="url(#e1)"/><rect x="440" y="40" width="190" height="80" rx="8" class="soft-p"/><text x="535" y="64" text-anchor="middle" class="sm">3. pekiştirmeli öğrenme (PPO)</text><text x="535" y="86" text-anchor="middle" class="sm">ödül r − β log π/π<tspan baseline-shift="-3" font-size="75%">SFT</tspan></text><text x="535" y="104" text-anchor="middle" class="sm">→ PPO ile en büyükle</text><rect x="20" y="150" width="600" height="30" rx="5" class="soft-b"/><text x="320.0" y="169.0" text-anchor="middle" class="sm">DPO: 2 ve 3 tek adımda — tercih çiftlerinde −log σ(β[log π/π<tspan baseline-shift="-3" font-size="75%">ref</tspan> (y<tspan baseline-shift="-3" font-size="75%">w</tspan>) − log π/π<tspan baseline-shift="-3" font-size="75%">ref</tspan> (y<tspan baseline-shift="-3" font-size="75%">l</tspan>)])</text><path d="M115,122 L115,148" class="ink dash" marker-end="url(#e4)"/><text x="122" y="140" class="sm">π<tspan baseline-shift="-3" font-size="75%">ref</tspan> = SFT modeli</text></svg><figcaption>Şekil 3 — InstructGPT tarifi (Ouyang ve ark. 2022). SFT modeli hem PPO'nun başlangıcı hem de KL cezasının referansıdır. DPO (Rafailov ve ark. 2023) KL cezalı amacın kapalı form çözümünü kullanarak ödül modelini ve RL döngüsünü ortadan kaldırır.</figcaption></figure>

:::tanım RLHF'nin üç adımı
1. **Denetimli ince ayar (SFT):** etiketleyicilerin istemlere yazdığı örnek yanıtlarla GPT-3'ü çapraz entropiyle ince ayarla.
2. **Ödül modeli (RM):** aynı isteme modelin ürettiği birkaç yanıtı etiketleyiciler iyiden kötüye sıralar. Bir model $r_\phi(x,y)$, tercih edilen yanıta ($y_w$) daha yüksek skor vermeyi öğrenir.
3. **Pekiştirmeli öğrenme:** politika $\pi_\theta$ (SFT'den başlatılır), PPO algoritmasıyla şu amacı en büyükler:
$$\max_\pi\ \mathbb E_{x,\,y\sim\pi}\big[r_\phi(x,y)\big]-\beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{SFT}}(\cdot\mid x)\big).$$
Pratikte her yanıtın ödülü $r_\phi(x,y)-\beta\log\frac{\pi(y\mid x)}{\pi_{\text{SFT}}(y\mid x)}$ olarak verilir. "PPO-ptx" çeşidi, genel dil yeteneğinin kaybolmaması için ön eğitim gradyanlarını da karıştırır.
:::

:::türetim Bradley–Terry ödül modeli
İki yanıttan $y_w$'nin $y_l$'ye tercih edilme olasılığı, gizli "kalite" skorlarının farkının lojistik fonksiyonu olarak modellenir:
$$P(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big),\qquad\mathcal L_{RM}=-\log\sigma\big(r(x,y_w)-r(x,y_l)\big).$$
Bu, iki yanıtın ödül farkı üzerinde lojistik regresyondur (M1.4). Ödüller yalnız **farkları** üzerinden tanımlıdır: her yanıta aynı sabiti eklemek kaybı değiştirmez. El hesabı: $r_w=2$, $r_l=0.5$ için $-\log\sigma(1.5)=\mathbf{0.201}$. KL cezalı ödül: $r=1$, $\beta=0.1$, $\log\frac{\pi}{\pi_{\text{SFT}}}=2$ için $1-0.1\cdot2=\mathbf{0.8}$. Politika referanstan ne kadar uzaklaşırsa ödülünden o kadar kaybeder.
:::

**KL cezası neden?** Ödül modeli kusurludur: insan tercihlerinin sonlu bir örneğinden öğrenilmiştir. Politika ödülü sınırsız en büyüklerse ödül modelinin hatalarını sömürür. Bu, ödül modeline göre yüksek ama insan gözünde saçma yanıtlar demektir: **ödül sömürüsü** (reward hacking). KL terimi politikayı ödül modelinin güvenilir olduğu bölgede, yani referansa yakın tutar.

Sonuç çarpıcıydı: insan değerlendiricilerin gözünde 1.3B parametreli InstructGPT'nin çıktıları, 100 kat büyük 175B GPT-3'ün çıktılarına **tercih edildi**. Hizalama, ölçekten ucuza gelmişti.

### ChatGPT, GPT-4 ve sonrası {#chatgpt}

InstructGPT kullanıcı komutlarını izliyordu ve bağlam kurmaya gerek kalmadan genel sorulara çok daha iyi cevap veriyordu. Ama sohbet sürdüremiyordu: her seferinde tek bir talimat. Bir sonraki adım **ChatGPT** oldu. ChatGPT, GPT-3.5 serisinden bir modelin InstructGPT'ye benzer biçimde (RLHF ile) ve sohbet verisiyle eğitilmiş hâlidir. GPT-3.5 metin ve kaynak kod üzerinde eğitilmişti; verisi Eylül 2021'de donduruldu. **GPT-4** Mart 2023'te çıktı. Eğitim verisi, eğitim stratejisi ve model boyutu açıklanmadı. ChatGPT-3.5'e göre daha iyi yanıtlar veriyordu ve **çok kipliydi** (görüntü girdisi kabul ediyordu).

:::uyarı Kaynak düzeltmesi: tarihler ve "GPT-5 olacak"
- ChatGPT Aralık 2022'de değil, **30 Kasım 2022**'de yayımlandı (modülün giriş metnindeki "Aralık" da düzeltildi; M4.7'deki zaman çizelgesi zaten Kasım diyor).
- Slaytın son sayfası GPT-5'i **gelecek zamanla** anlatıyor ("ChatGPT 5 ... olacak"). Bu bilgi eskidir: OpenAI GPT-5'i **Ağustos 2025**'te yayımladı. Aynı ders setindeki ch_16 zaman çizelgesi 2026 için GPT-5.5'i bile listeliyor. Bu sayfa, slaytların yazıldığı dönemin beklentisi olarak okunmalı.
- Zaman çizelgesindeki diğer düzeltmeler (GPT-1/2/3 adları, GPT-3.5'in açıklanmamış parametre sayısı, Mistral ve Qwen'in 2023 çıkışı, Gemma 3'ün Mart 2025 tarihi, "Antropic" ve "Opes" yazımları) M4.7'de.
:::

Bu alandaki model adları ve sürümleri çok hızlı eskir. Bu sayfadaki teknik içerik (ön eğitim amaçları, RLHF, DPO) kalıcıdır; ürün adları ve tarihler için şirketlerin kendi duyurularına bakmak gerekir.

### DPO: ödül modeli ve RL olmadan tercih öğrenmek {#dpo}

PPO ile RLHF karmaşıktır: dört model (politika, referans, ödül, değer), örnekleme döngüsü ve hassas hiperparametreler gerekir. **DPO** (Direct Preference Optimization; Rafailov ve ark., NeurIPS 2023), KL cezalı amacın optimumunun kapalı formda yazılabileceğini fark eder.

:::türetim KL cezalı amacın optimumu
Sabit bir $x$ için amaç $J(\pi)=\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}$ olsun. $Z=\sum_y\pi_{\text{ref}}(y)e^{r(y)/\beta}$ ve $\pi^*(y)=\pi_{\text{ref}}(y)e^{r(y)/\beta}/Z$ tanımlayalım. O zaman
$$J(\pi)=-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)e^{r(y)/\beta}}=-\beta\,\mathrm{KL}(\pi\,\|\,\pi^*)+\beta\log Z .$$
$\beta\log Z$, $\pi$'den bağımsızdır ve KL $\ge0$'dır. Dolayısıyla $J$'nin tek maksimumu $\pi=\pi^*$'dır:
$$\pi^*(y)\propto\pi_{\text{ref}}(y)\,e^{r(y)/\beta}.$$
**Koşul:** bu, politika tüm olası $\pi$'leri temsil edebiliyorsa geçerlidir (tablo politikası ya da yeterince esnek bir ağ).
:::

:::türetim DPO kaybı
Optimum formülü ters çevrilince ödül politikayla ifade edilir: $r(y)=\beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)}+\beta\log Z$. Bunu Bradley–Terry olabilirliğine koyalım. Farkta $\beta\log Z$ **sadeleşir**; bölüşüm fonksiyonunu hesaplamak gerekmez. $\pi^*$ yerine eğitilen politika $\pi_\theta$ konunca kayıp şu olur:
$$\mathcal L_{DPO}=-\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\Big).$$
Bu, yalnız tercih çiftleri üzerinde sıradan bir sınıflandırma kaybıdır: ödül modeli yok, örnekleme yok, RL yok. Politika kendi "örtük ödülünü" $\hat r=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ taşır. El hesabı ($\beta=0.1$, log oranlar $2$ ve $-1$): $-\log\sigma(0.1\cdot3)=-\log\sigma(0.3)=\mathbf{0.554}$. Gradyanın ağırlığı $\sigma(-0.3)=0.43$'tür: örtük ödül sırayı ne kadar **yanlış** tahmin ediyorsa güncelleme o kadar büyüktür.
:::

**Best-of-n.** En basit hizalama yöntemi hiç eğitim içermez: referanstan $n$ yanıt örnekle, ödül modeline göre en iyisini seç. Bu yöntemin dağılımı kapalı formdadır (ek x1). Referanstan uzaklığı da sınırlıdır: $\mathrm{KL}\le\log n-\frac{n-1}{n}$. Bedeli, her istem için $n$ kat çıkarımdır.

Aşağıdaki deneyde $K$ olası yanıtlı tam izlenebilir bir oyuncakta KL cezalı optimumu, best-of-n'i ve sonlu sayıda tercih çiftiyle eğitilen DPO'yu ödül–KL düzleminde karşılaştır.

::lab l3

### Bağlamsal gömmeleri kullanmanın yolları {#kullanim}

| yol | ne eğitilir | örnek |
|---|---|---|
| öznitelik çıkarma | yalnız görev modeli; ön eğitilmiş ağ dondurulur | ELMo; BERT'in [CLS] vektörüyle lojistik regresyon (doğrusal sonda) |
| tam ince ayar | tüm ağ + küçük bir baş | GPT-1, BERT |
| istemle kullanma | hiçbir şey; görev istemle anlatılır | GPT-2/3 sıfır/az örnekli |
| talimat ayarı + tercih öğrenme | tüm ağ (SFT, RLHF, DPO) | InstructGPT, ChatGPT |
| parametre-verimli ince ayar | ağın küçük bir eki (ör. düşük ranklı LoRA matrisleri) | büyük modelleri tek GPU'da uyarlamak (Faz 6–8) |

Aşama d bu tablonun ilk iki satırını sentetik bir dilde karşılaştırıyor. Küçük bir BERT'i MLM ile ön eğitip az etiketle ince ayarlamak, aynı modeli sıfırdan eğitmekle karşılaştırılıyor; ayrıca yalnız doğrusal sonda ile tam ince ayar karşılaştırılıyor. Aşama e, DPO'nun tam izlenebilir bir oyuncakta analitik optimum $\pi^*$'a yakınsadığını gösteriyor.

### PyTorch ve Hugging Face'te {#pytorch}

| kavram | kod | not |
|---|---|---|
| önceden eğitilmiş model | `AutoModel.from_pretrained("bert-base-uncased")` | `transformers` |
| MLM başı | `AutoModelForMaskedLM`; `pipeline("fill-mask")` | |
| MLM maskeleme | `DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)` | 80/10/10 kuralı içeride |
| sınıflandırma ince ayarı | `AutoModelForSequenceClassification(..., num_labels=K)` | [CLS] + doğrusal |
| nedensel LM | `AutoModelForCausalLM.from_pretrained("gpt2")`; `model.generate(...)` | M4.7 |
| dizi log olasılığı | `logits.log_softmax(-1).gather(-1, y[..., None])` × yanıt maskesi | DPO için (aşama c) |
| Bradley–Terry | `-F.logsigmoid(r_w - r_l).mean()` | |
| DPO | `-F.logsigmoid(beta * ((lp_w - ref_w) - (lp_l - ref_l))).mean()` | `trl.DPOTrainer` hazır uygulama |
| dondurma | `for p in model.base_model.parameters(): p.requires_grad_(False)` | doğrusal sonda |

### Özet {#ozet}

- Sabit gömme her kelimeye tek vektör verir; bağlamsal gömme, cümlenin tamamına bakan bir ağın o kelimedeki çıktısıdır. Ortak tarif: kendinden gözetimli ön eğitim + aktarım.
- ELMo (2018): karakter CNN + 2 katmanlı biLSTM (4096/512, artık bağlantı); temsil $\gamma\sum_js_j\mathbf h_{k,j}$; iki yön ayrı ("sığ" çift yönlülük). Slayt düzeltmesi: girdi word2vec değil, karakter CNN'i.
- GPT-1 (2018): kod çözücü, ileri LM, BooksCorpus; 12/768/12, ≈ 117M (slayttaki 110M BERT-base'in sayısı); son token + doğrusal katman, tüm ağ ince ayarlanır, yardımcı LM kaybı λ = 0.5; görevler token dizisine çevrilir.
- BERT (2018/2019): kodlayıcı; çift yönlü dikkatte sonraki token amacı kopyalamaya dönüşür, bu yüzden MLM (%15; 80/10/10; 512'de 76.8 seçim) + NSP. Base 109 482 240 ≈ 110M, large 340M; GLUE 80.5. RoBERTa: NSP gereksiz.
- BART (2019): gürültü giderici kodlayıcı–kod çözücü; token maskeleme/silme, metin doldurma (Poisson λ = 3), cümle karıştırma, belge döndürme.
- GPT-2 (1.5B, WebText 40 GB) istemle sıfır örnekli; GPT-3 (175B, 300B token) bağlam içi öğrenme. Parametre ≈ $12Ld^2$.
- RLHF: SFT → Bradley–Terry ödül modeli ($-\log\sigma(r_w-r_l)$) → KL cezalı PPO. 1.3B InstructGPT > 175B GPT-3 (insan tercihi). KL, ödül sömürüsünü sınırlar.
- KL cezalı amacın optimumu $\pi^*\propto\pi_{\text{ref}}e^{r/\beta}$. DPO bunu ters çevirip Bradley–Terry'ye koyar, $Z$ sadeleşir: $-\log\sigma(\beta\Delta\log\frac{\pi}{\pi_{\text{ref}}})$. Best-of-n: KL ≤ log n − (n−1)/n.
- Tarih düzeltmeleri: ChatGPT 30 Kasım 2022; GPT-5 Ağustos 2025'te çıktı (slayt gelecek zamanlı).
