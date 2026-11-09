### Problem: milyonlarca parametre, sınırlı veri {#problem}

Derin ağların on binlerce, hatta milyonlarca parametresi vardır. Bu özgürlük karmaşık örüntüleri öğrenmelerini sağlar ama aynı zamanda eğitim setini — gürültüsüyle birlikte — **ezberlemelerine** de izin verir (aşırı uyum, M1.3). Bu modülde sinir ağlarına özgü düzenlileştirme (regularization) araçlarını sırayla ele alıyoruz:

1. **Early stopping** — doğrulama hatası artmaya başlayınca durmak. İlk ve çoğu zaman en etkili savunma.
2. **L1 / L2 düzenlileştirmesi ve weight decay** — ağırlıkların büyüklüğünü cezalandırmak.
3. **Dropout** (ve türevleri: alpha dropout, MC dropout) — her adımda rastgele nöronları kapatmak.
4. **Max-norm** — her nöronun ağırlık vektörünün normunu sınırlamak.

Ek olarak: M3.6'da gördüğümüz **BatchNorm** kararsız gradyanlar için tasarlanmış olsa da batch gürültüsü sayesinde iyi bir düzenlileştiricidir; ve patlayan gradyanlara karşı **gradyan kırpma** (düzenlileştirme değil, kararlılık aracı) bu modülde.

Çoğu düzenlileştirme $\text{düzenlileştirilmiş kayıp}=\text{kayıp}+\text{ceza terimi}$ biçimindedir (L1/L2). Ama hepsi değil: dropout eğitim **sürecine** gürültü ekler, max-norm bir **kısıt**tır (her adımdan sonra izdüşüm), early stopping ise eğitimin **süresini** sınırlar. Ortak noktaları: modelin eğitim verisine tam uymasını zorlaştırıp daha basit, daha iyi genelleşen çözümlere yöneltmeleri.

### Early stopping {#early}

Eğitim ilerledikçe eğitim hatası neredeyse hep azalır; doğrulama hatası ise önce azalır, sonra (model gürültüyü ezberlemeye başlayınca) artar. **Early stopping:** doğrulama hatasını düzenli aralıklarla ölç, en iyi değeri ve o anki ağırlıkları sakla; `patience` kadar ölçüm boyunca iyileşme olmazsa dur ve **en iyi ağırlıkları geri yükle**.

:::örnek Patience ile karar
Doğrulama kayıpları (epoch 0'dan): $[0.9,\,0.7,\,0.6,\,0.62,\,0.65,\,0.61]$, patience 2. En iyi epoch 2 (0.6). Epoch 3 ve 4 iyileştirmiyor → epoch 4'ün sonunda dur, epoch 2'nin ağırlıklarını geri yükle. Epoch 5'teki 0.61 hiç görülmez — patience'ın bedeli bu: çok küçük patience gürültülü bir doğrulama eğrisinde erken pes eder, çok büyük patience boşa hesap harcar.
:::

Early stopping neden bir düzenlileştiricidir? Sezgi: sıfıra yakın başlayan ağırlıklar her adımda biraz büyür; erken durmak ağırlıkların gidebileceği mesafeyi sınırlar. Basit bir durumda bu sezgi kesinleşir:

:::türetim Kuadratik kayıpta early stopping ≈ L2
**Varsayımlar:** kayıp, optimum $w^*$ civarında kuadratik: $L(w)=\frac12(w-w^*)^\top H(w-w^*)$, $H$'nin özdeğerleri $\lambda_i>0$; GD $w_0=0$'dan başlar, $\eta\lambda_i<1$.

GD adımı $w_{t+1}=w_t-\eta H(w_t-w^*)$. $H$'nin özvektör tabanında her bileşen ayrı evrilir: $w_{t,i}-w^*_i=(1-\eta\lambda_i)(w_{t-1,i}-w^*_i)$, dolayısıyla
$$w_{t,i}=\big[1-(1-\eta\lambda_i)^t\big]\,w^*_i .$$
L2 cezalı ($\frac\alpha2\|w\|^2$) problemin çözümü aynı tabanda (M1.3, ridge): $\tilde w_i=\dfrac{\lambda_i}{\lambda_i+\alpha}w^*_i$.

İkisi de büyük-$\lambda_i$ (dik) yönleri neredeyse tam öğrenir, küçük-$\lambda_i$ (düz) yönleri bastırır. Küçük $\eta\lambda_i$ için $(1-\eta\lambda_i)^t\approx e^{-t\eta\lambda_i}$, ve $\lambda_i\ll\alpha$ iken iki küçültme çarpanı $\approx t\eta\lambda_i$ ile $\approx\lambda_i/\alpha$ olur; eşleştirme:
$$\alpha\approx\frac1{\eta\,t}.$$
Yani $t$ adımda durmak, yaklaşık $\alpha=1/(\eta t)$ güçlü bir L2 cezasına denktir (Goodfellow ve ark. 2016, §7.8). Bu bir **yaklaşık eşdeğerliktir** (iki çarpanın şekli farklıdır) ve yalnız kuadratik kayıp + sıfır başlangıç için kesinleşir; derin ağlarda bir sezgi olarak kalır.
:::

Avantajı: hiçbir hiperparametreyi değiştirmeden, tek bir eğitimde "düzenlileştirme gücünü" taramış olursun (her adım sayısı ayrı bir $\alpha$). Bedeli: ayrı bir doğrulama seti gerekir, ve doğrulama setine göre seçim yaptığın için o setteki performans iyimser bir tahmindir (son raporu ayrı bir test setinde ver, M3.9).

### L1 ve L2 düzenlileştirmesi, weight decay {#l1l2}

M1.3'teki cezalar sinir ağlarında da aynıdır: $L+\frac\lambda2\sum\|W^l\|_F^2$ (L2) ya da $L+\lambda\sum|w|$ (L1). Gradyana katkıları sırasıyla $\lambda w$ ve $\lambda\,\mathrm{sign}(w)$.

- **L2** her adımda ağırlığı orantılı olarak küçültür; SGD'de $w\leftarrow w-\eta(\nabla L+\lambda w)=(1-\eta\lambda)w-\eta\nabla L$: **weight decay** adı buradan gelir.
- **L1** ağırlıkları tam sıfıra iter: **seyrek** (sparse) bir model gerekiyorsa L1 kullan ve eğitim sonunda çok küçük ağırlıkları sıfırla.
- **Adam ile fark (M3.4):** uyarlamalı optimizer'larda L2 cezasının gradyanı da $\sqrt{\hat v}$'ye bölünür; büyük gradyanlı parametreler daha az küçülür. **AdamW** küçültmeyi gradyandan ayırır: $w\leftarrow w-\eta\lambda w-\eta\,\hat m/(\sqrt{\hat v}+\varepsilon)$. Adam kullanıyorsan L2 değil AdamW'nin `weight_decay`'ini kullan.
- **Hangi parametreler?** Genelde yalnız ağırlık matrisleri; bias'lar ve normalizasyon katmanlarının $\gamma,\beta$'sı muaf tutulur (PyTorch'ta optimizer'a iki parametre grubu verilir). BN'li ağlarda weight decay'in asıl etkisinin etkin öğrenme oranı olduğunu M3.6'da gördük.

### Dropout {#dropout}

Dropout (Hinton ve ark. 2012; Srivastava ve ark. 2014) derin ağlar için en popüler düzenlileştirme tekniklerinden biridir. Etkisi küçük görünebilir — en iyi ağlarda bile tipik olarak %1–2 doğruluk artışı — ama doğruluk zaten %95 iken %2 kazanç hata oranını %5'ten ≈ %3'e, yani **≈ %40** azaltmak demektir.

:::tanım Dropout algoritması
Her eğitim adımında her nöron (girdi nöronları dahil, **çıktı nöronları hariç**), diğerlerinden bağımsız olarak $p$ olasılıkla geçici olarak **düşürülür**: o adımda çıktısı 0'dır, tamamen yok sayılır; bir sonraki adımda yeniden etkin olabilir. $p$'ye **dropout oranı** denir. Eğitimden sonra hiçbir nöron düşürülmez.
:::

<figure class="fig wide"><svg viewBox="0 0 640 245" role="img" aria-label="Solda tüm nöronları bağlı bir ağ; sağda bazı girdi ve gizli nöronları çarpı ile düşürülmüş, bağlantıları silinmiş aynı ağ"><line x1="40" y1="60.0" x2="120" y2="56.0" class="mute"/><line x1="40" y1="60.0" x2="120" y2="88.0" class="mute"/><line x1="40" y1="60.0" x2="120" y2="120.0" class="mute"/><line x1="40" y1="60.0" x2="120" y2="152.0" class="mute"/><line x1="40" y1="60.0" x2="120" y2="184.0" class="mute"/><line x1="40" y1="100.0" x2="120" y2="56.0" class="mute"/><line x1="40" y1="100.0" x2="120" y2="88.0" class="mute"/><line x1="40" y1="100.0" x2="120" y2="120.0" class="mute"/><line x1="40" y1="100.0" x2="120" y2="152.0" class="mute"/><line x1="40" y1="100.0" x2="120" y2="184.0" class="mute"/><line x1="40" y1="140.0" x2="120" y2="56.0" class="mute"/><line x1="40" y1="140.0" x2="120" y2="88.0" class="mute"/><line x1="40" y1="140.0" x2="120" y2="120.0" class="mute"/><line x1="40" y1="140.0" x2="120" y2="152.0" class="mute"/><line x1="40" y1="140.0" x2="120" y2="184.0" class="mute"/><line x1="40" y1="180.0" x2="120" y2="56.0" class="mute"/><line x1="40" y1="180.0" x2="120" y2="88.0" class="mute"/><line x1="40" y1="180.0" x2="120" y2="120.0" class="mute"/><line x1="40" y1="180.0" x2="120" y2="152.0" class="mute"/><line x1="40" y1="180.0" x2="120" y2="184.0" class="mute"/><line x1="120" y1="56.0" x2="200" y2="56.0" class="mute"/><line x1="120" y1="56.0" x2="200" y2="88.0" class="mute"/><line x1="120" y1="56.0" x2="200" y2="120.0" class="mute"/><line x1="120" y1="56.0" x2="200" y2="152.0" class="mute"/><line x1="120" y1="56.0" x2="200" y2="184.0" class="mute"/><line x1="120" y1="88.0" x2="200" y2="56.0" class="mute"/><line x1="120" y1="88.0" x2="200" y2="88.0" class="mute"/><line x1="120" y1="88.0" x2="200" y2="120.0" class="mute"/><line x1="120" y1="88.0" x2="200" y2="152.0" class="mute"/><line x1="120" y1="88.0" x2="200" y2="184.0" class="mute"/><line x1="120" y1="120.0" x2="200" y2="56.0" class="mute"/><line x1="120" y1="120.0" x2="200" y2="88.0" class="mute"/><line x1="120" y1="120.0" x2="200" y2="120.0" class="mute"/><line x1="120" y1="120.0" x2="200" y2="152.0" class="mute"/><line x1="120" y1="120.0" x2="200" y2="184.0" class="mute"/><line x1="120" y1="152.0" x2="200" y2="56.0" class="mute"/><line x1="120" y1="152.0" x2="200" y2="88.0" class="mute"/><line x1="120" y1="152.0" x2="200" y2="120.0" class="mute"/><line x1="120" y1="152.0" x2="200" y2="152.0" class="mute"/><line x1="120" y1="152.0" x2="200" y2="184.0" class="mute"/><line x1="120" y1="184.0" x2="200" y2="56.0" class="mute"/><line x1="120" y1="184.0" x2="200" y2="88.0" class="mute"/><line x1="120" y1="184.0" x2="200" y2="120.0" class="mute"/><line x1="120" y1="184.0" x2="200" y2="152.0" class="mute"/><line x1="120" y1="184.0" x2="200" y2="184.0" class="mute"/><line x1="200" y1="56.0" x2="280" y2="80.0" class="mute"/><line x1="200" y1="56.0" x2="280" y2="160.0" class="mute"/><line x1="200" y1="88.0" x2="280" y2="80.0" class="mute"/><line x1="200" y1="88.0" x2="280" y2="160.0" class="mute"/><line x1="200" y1="120.0" x2="280" y2="80.0" class="mute"/><line x1="200" y1="120.0" x2="280" y2="160.0" class="mute"/><line x1="200" y1="152.0" x2="280" y2="80.0" class="mute"/><line x1="200" y1="152.0" x2="280" y2="160.0" class="mute"/><line x1="200" y1="184.0" x2="280" y2="80.0" class="mute"/><line x1="200" y1="184.0" x2="280" y2="160.0" class="mute"/><circle cx="40" cy="60.0" r="11" class="ca w2"/><circle cx="40" cy="100.0" r="11" class="ca w2"/><circle cx="40" cy="140.0" r="11" class="ca w2"/><circle cx="40" cy="180.0" r="11" class="ca w2"/><circle cx="120" cy="56.0" r="11" class="ink w2"/><circle cx="120" cy="88.0" r="11" class="ink w2"/><circle cx="120" cy="120.0" r="11" class="ink w2"/><circle cx="120" cy="152.0" r="11" class="ink w2"/><circle cx="120" cy="184.0" r="11" class="ink w2"/><circle cx="200" cy="56.0" r="11" class="ink w2"/><circle cx="200" cy="88.0" r="11" class="ink w2"/><circle cx="200" cy="120.0" r="11" class="ink w2"/><circle cx="200" cy="152.0" r="11" class="ink w2"/><circle cx="200" cy="184.0" r="11" class="ink w2"/><circle cx="280" cy="80.0" r="11" class="cp w2"/><circle cx="280" cy="160.0" r="11" class="cp w2"/><line x1="360" y1="60.0" x2="440" y2="88.0" class="mute"/><line x1="360" y1="60.0" x2="440" y2="120.0" class="mute"/><line x1="360" y1="60.0" x2="440" y2="184.0" class="mute"/><line x1="360" y1="100.0" x2="440" y2="88.0" class="mute"/><line x1="360" y1="100.0" x2="440" y2="120.0" class="mute"/><line x1="360" y1="100.0" x2="440" y2="184.0" class="mute"/><line x1="360" y1="180.0" x2="440" y2="88.0" class="mute"/><line x1="360" y1="180.0" x2="440" y2="120.0" class="mute"/><line x1="360" y1="180.0" x2="440" y2="184.0" class="mute"/><line x1="440" y1="88.0" x2="520" y2="56.0" class="mute"/><line x1="440" y1="88.0" x2="520" y2="152.0" class="mute"/><line x1="440" y1="120.0" x2="520" y2="56.0" class="mute"/><line x1="440" y1="120.0" x2="520" y2="152.0" class="mute"/><line x1="440" y1="184.0" x2="520" y2="56.0" class="mute"/><line x1="440" y1="184.0" x2="520" y2="152.0" class="mute"/><line x1="520" y1="56.0" x2="600" y2="80.0" class="mute"/><line x1="520" y1="56.0" x2="600" y2="160.0" class="mute"/><line x1="520" y1="152.0" x2="600" y2="80.0" class="mute"/><line x1="520" y1="152.0" x2="600" y2="160.0" class="mute"/><circle cx="360" cy="60.0" r="11" class="ca w2"/><circle cx="360" cy="100.0" r="11" class="ca w2"/><circle cx="360" cy="140.0" r="11" class="mute dash"/><path d="M354 134.0 l12 12 M366 134.0 l-12 12" class="cq w2"/><circle cx="360" cy="180.0" r="11" class="ca w2"/><circle cx="440" cy="56.0" r="11" class="mute dash"/><path d="M434 50.0 l12 12 M446 50.0 l-12 12" class="cq w2"/><circle cx="440" cy="88.0" r="11" class="ink w2"/><circle cx="440" cy="120.0" r="11" class="ink w2"/><circle cx="440" cy="152.0" r="11" class="mute dash"/><path d="M434 146.0 l12 12 M446 146.0 l-12 12" class="cq w2"/><circle cx="440" cy="184.0" r="11" class="ink w2"/><circle cx="520" cy="56.0" r="11" class="ink w2"/><circle cx="520" cy="88.0" r="11" class="mute dash"/><path d="M514 82.0 l12 12 M526 82.0 l-12 12" class="cq w2"/><circle cx="520" cy="120.0" r="11" class="mute dash"/><path d="M514 114.0 l12 12 M526 114.0 l-12 12" class="cq w2"/><circle cx="520" cy="152.0" r="11" class="ink w2"/><circle cx="520" cy="184.0" r="11" class="mute dash"/><path d="M514 178.0 l12 12 M526 178.0 l-12 12" class="cq w2"/><circle cx="600" cy="80.0" r="11" class="cp w2"/><circle cx="600" cy="160.0" r="11" class="cp w2"/><text x="150" y="232" text-anchor="middle" class="sm">(a) tam ağ — çıkarımda</text><text x="470" y="232" text-anchor="middle" class="sm">(b) bir eğitim adımında inceltilmiş ağ</text><text x="40" y="22" text-anchor="middle" class="ta">girdi</text><text x="280" y="22" text-anchor="middle" class="tp">çıktı</text></svg><figcaption>Şekil 1 — Dropout. Her eğitim adımında her girdi ve gizli nöron p olasılıkla (bağımsız olarak) geçici olarak düşürülür: çıktısı 0 olur, bağlantıları o adımda yokmuş gibi davranır. Çıktı nöronları hiçbir zaman düşürülmez. Bir sonraki adımda başka bir rastgele alt ağ seçilir; çıkarımda tüm ağ kullanılır (Srivastava ve ark. 2014, Şekil 1'in yeniden çizimi).</figcaption></figure>

**Oran seçimi.** $p$ tipik olarak %10–50: RNN'lerde %20–30'a, CNN'lerde %40–50'ye yakın; Srivastava ve ark. girdi katmanında %20, gizli katmanlarda %50 kullandı. Pratikte dropout'u çoğu zaman yalnız üstteki bir–üç katmana (çıktı katmanı hariç) uygulamak yeterlidir. Model aşırı uyuyorsa $p$'yi artır, eksik uyuyorsa azalt; büyük katmanlarda büyük, küçüklerde küçük $p$.

**Ölçekleme — önemli teknik ayrıntı.** $p=0.5$ olsun. Eğitimde bir nöron girdilerinin ortalama yarısını görür; testte hepsini görür → toplam girdi sinyali yaklaşık **iki kat** büyük olur ve ağ eğitildiği koşullardan çıkar. Bunu telafi etmenin iki eşdeğer yolu vardır:

| yöntem | eğitimde | testte |
|---|---|---|
| klasik (orijinal makale) | düşür, ölçekleme yok | ağırlıkları **tutma olasılığı** $(1-p)$ ile çarp |
| **inverted dropout** (PyTorch, Keras) | düşür, kalanları $\frac1{1-p}$ ile çarp | hiçbir şey yapma |

:::türetim Inverted dropout beklenen değeri korur
Maske $m\sim\mathrm{Bernoulli}(1-p)$ (1 = tut), $y=\frac{m}{1-p}\,x$. Beklenen değer: $\mathbb E[y]=\frac{\mathbb E[m]}{1-p}x=\frac{1-p}{1-p}x=x$ — test zamanındaki (dropout'suz) çıktıya eşit. Varyans: $\mathrm{Var}[y]=\frac{x^2\,\mathrm{Var}[m]}{(1-p)^2}=\frac{p}{1-p}x^2$: dropout bir **çarpımsal gürültü**dür; $p=0.5$'te gürültünün std'si sinyalin kendisi kadardır.

Geri geçiş: $y=x\odot m/(1-p)$ ⇒ $\partial L/\partial x=\partial L/\partial y\odot m/(1-p)$ — ileri geçişteki **aynı** maske. Düşürülen nöronlar o adımda gradyan almaz.
:::

Inverted dropout'un üstünlüğü pratiktir: test kodu değişmez (dropout yalnız `model.train()` modunda etkin, `model.eval()`'de kimlik fonksiyonu).

:::türetim Doğrusal modelde dropout = veriye uyarlanmış L2
Doğrusal regresyon, girdilere inverted dropout: $\hat y=\sum_j w_jx_j\frac{m_j}{1-p}$. Maskeler üzerinden beklenen kare hata (yukarıdaki varyans hesabıyla, bağımsız $m_j$):
$$\mathbb E_m\big[(t-\hat y)^2\big]=(t-w^\top x)^2+\mathrm{Var}[\hat y]=(t-w^\top x)^2+\frac{p}{1-p}\sum_j w_j^2x_j^2 .$$
Veri üzerinden toplayınca ceza $\frac p{1-p}\sum_j w_j^2\big(\sum_i x_{ij}^2\big)$: her ağırlık, kendi özelliğinin ölçeğiyle ağırlıklandırılmış bir L2 cezası alır (Wager ve ark. 2013). Derin ağlarda böyle kapalı bir eşdeğerlik yoktur; dropout orada daha zengin bir gürültü düzenlileştiricisidir.
:::

**Neden işe yarıyor? İki içgörü.**

1. **Ortak uyum (co-adaptation) yok.** Dropout'la eğitilen bir nöron komşularına güvenemez — her an düşürülebilirler; kendi başına olabildiğince yararlı olmak zorundadır. Birkaç girdi nöronuna aşırı da yaslanamaz; tüm girdilerine dikkat etmelidir. Sonuç: girdilerdeki küçük değişimlere daha az duyarlı, daha sağlam ve daha iyi genelleşen bir ağ.
2. **Örtük topluluk.** Her nöron ya vardır ya yoktur: $N$ düşürülebilir nöronla $2^N$ olası alt ağ vardır. Bu sayı o kadar büyüktür ki aynı alt ağın iki kez örneklenmesi pratikte imkânsızdır; 10 000 eğitim adımı, her biri tek bir adım (bir mini-batch) görmüş 10 000 farklı ağ eğitmek demektir. Ağırlıkları paylaştıkları için bağımsız değillerdir, ama yine de farklıdırlar; sonuçtaki ağ bu küçük ağların **ortalamasını alan bir topluluk** (M2.5) gibi davranır. Test zamanındaki ağırlık ölçekleme bu ortalamanın ucuz bir yaklaşımıdır: tek gizli katmanlı, softmax çıkışlı bir ağda $2^N$ alt ağın tahminlerinin **geometrik ortalamasını** tam olarak verir (Hinton ve ark. 2012); daha derin ağlarda yalnız yaklaşıktır.

:::uyarı Dropout ve BatchNorm
Dropout bir katmanın çıktısının varyansını eğitimde ve testte farklı yapar ($\frac p{1-p}x^2$ ek varyans yalnız eğitimde). Ardından gelen bir BN, koşu istatistiklerini eğitimdeki (gürültülü) varyansla öğrenir ve testte uyumsuz kalır ("varyans kayması", Li ve ark. 2019). Pratik kural: dropout'u **son BN katmanından sonra** (örneğin sınıflandırıcı başında) kullan.
:::

### Alpha dropout ve MC dropout {#alpha}

**Alpha dropout.** SELU ile öz-normalize bir ağ (M3.5) kurduysan sıradan dropout öz-normalizasyonu bozar: sıfır, SELU'nun "doğal" değeri değildir ve ortalama/varyans kayar. Klambauer ve ark. (SELU'yu tanıtan aynı makalede) **alpha dropout**'u önerdi: girdilerin ortalamasını ve std'sini koruyan bir dropout.

:::ayrıntı Alpha dropout nasıl çalışır?
Düşürülen birimler 0 yerine SELU'nun negatif doyma değerine, $\alpha'=-\lambda\alpha\approx-1.7581$'e eşitlenir. $q=1-p$ tutma olasılığı, $d\sim\mathrm{Bernoulli}(q)$, girdi $x$ ortalama 0 varyans 1 iken $z=x\,d+\alpha'(1-d)$'nin ortalaması $(1-q)\alpha'$, varyansı $q+q(1-q)\alpha'^2$ olur. Bir afin düzeltme $a z+b$ ile
$$a=\big(q+\alpha'^2q(1-q)\big)^{-1/2},\qquad b=-a(1-q)\alpha'$$
seçilince çıktı yeniden ortalama 0, varyans 1 olur. x1 ekinde bunu yazıp sıradan dropout'la karşılaştırıyorsun (`nn.AlphaDropout`).
:::

**Monte Carlo (MC) dropout** (Gal ve Ghahramani 2016). Örtük topluluk içgörüsünü **çıkarımda** da kullan: aynı test girdisi için dropout **açıkken** $T$ tahmin yap (her biri farklı bir alt ağ), sonuçları üst üste koy ve ortalamasını al.

- Ortalama, dropout kapalı tek bir tahminden genellikle daha güvenilirdir.
- Tahminlerin **dağılımı** (std, sınıflandırmada entropi ya da sınıf olasılıklarının std'si) modelin **belirsizliği** hakkında bilgi verir: eğitim verisinden uzak girdilerde tahminler daha çok dağılır.
- $T$ (ör. 100) ayarlanabilir bir hiperparametredir: büyüdükçe tahminler ve belirsizlik kestirimleri daha isabetli olur, ama $T$'yi ikiye katlamak çıkarım süresini de ikiye katlar; belli bir sayıdan sonra kazanç azalır. MC dropout esasen bir **çıkarım** tekniğidir — gecikme (latency) ile doğruluk arasındaki ödünleşimi uygulamaya göre seç.

:::uyarı MC dropout uygulama tuzağı
Modelde eğitimde farklı davranan başka katmanlar (BatchNorm) varsa tüm modeli eğitim moduna alma (`model.train()`): BN batch istatistiklerine döner. Bunun yerine `model.eval()` yapıp **yalnız dropout modüllerini** eğitim moduna al (ya da dropout'u her zaman açık olan kendi modülünle değiştir; c ve d aşamaları). MC dropout'un verdiği belirsizlik bir **yaklaşımdır**; kalibre edilmiş bir güven aralığı değildir.
:::

Aşağıdaki deneyde 14 gürültülü noktaya 64 nöronlu bir ağ uyduruluyor. Önce hiçbir düzenlileştirme olmadan bak: eğitim hatası gürültü tabanının ($\sigma^2$) çok altına iner — ağ gürültüyü ezberliyor. Sonra dropout'u, weight decay'i aç ve early stopping'in (turuncu) seçtiği ağı son ağla (mavi) karşılaştır. Dropout açıkken mor çizgiler MC dropout bandıdır.

::lab l1

### Max-norm {#maxnorm}

Max-norm düzenlileştirmesi her nöronun **gelen** ağırlık vektörünü sınırlar: $\|w\|_2\le r$. Kayıba ceza eklemez; her eğitim adımından sonra ağırlığı izdüşürür:
$$w\leftarrow\begin{cases}w,&\|w\|_2\le r\\[2pt] w\,\dfrac{r}{\|w\|_2},&\|w\|_2>r.\end{cases}$$

<figure class="fig"><svg viewBox="0 0 300 220" role="img" aria-label="Yarıçapı r olan bir daire; dışarıdaki w vektörü daire üzerine radyal olarak izdüşürülüyor, içerideki u değişmiyor"><defs><marker id="mn1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker><marker id="mn2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cp-f"/></marker><marker id="mn3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><circle cx="130" cy="115" r="70" class="soft-p"/><circle cx="130" cy="115" r="70" class="cp dash"/><line x1="130" y1="115" x2="258" y2="46" class="ca w2 dash" marker-end="url(#mn1)"/><line x1="130" y1="115" x2="190.6" y2="82.3" class="cp w3" marker-end="url(#mn2)"/><line x1="130" y1="115" x2="91" y2="85" class="cb w2" marker-end="url(#mn3)"/><text x="262" y="44" class="ta">w</text><text x="176" y="110" class="tp">w·r/‖w‖</text><text x="70" y="80" class="tb">u</text><line x1="130" y1="115" x2="130" y2="185" class="mute"/><text x="135" y="160" class="sm">r</text><text x="20" y="208" class="sm">‖w‖ &gt; r: radyal olarak daireye çekilir; ‖u‖ ≤ r: değişmez</text></svg><figcaption>Şekil 2 — Max-norm bir izdüşümdür. Yön korunur, yalnız boy kesilir — norma göre gradyan kırpmanın (aşağıda) ağırlıklara uygulanmış hâli.</figcaption></figure>

$r$ küçüldükçe düzenlileştirme güçlenir ve aşırı uyum azalır; Srivastava ve ark. dropout ile birlikte $r\approx3$–4 kullandı (büyük öğrenme oranlarında ağırlıkların patlamasını önler). BN kullanılmayan ağlarda kararsız gradyanları hafifletmeye de yardım eder. PyTorch'ta hazır bir katman yoktur; `optimizer.step()`'ten sonra `torch.no_grad()` içinde `W.copy_(torch.renorm(W, p=2, dim=0, maxnorm=r))` (`nn.Linear`'ın ağırlığı (çıkış, giriş) shape'inde: her satır bir nöronun gelen ağırlıkları). x2 ekinde bunu yazıyorsun.

### Gradyan kırpma {#kirpma}

Patlayan gradyanlara karşı bir başka popüler teknik, geri yayılım sırasında gradyanları bir eşiği asla aşmayacak biçimde **kırpmaktır** (gradient clipping). En çok RNN'lerde kullanılır, çünkü BatchNorm'u RNN'lerde kullanmak zordur (M3.6); Transformer eğitiminde de standarttır (tipik eşik: global norm 1.0). Diğer ağ türlerinde çoğu zaman BN yeterlidir.

İki çeşidi vardır:

- **Değere göre:** her bileşen $[-c,c]$ aralığına kesilir: $g_i\leftarrow\min(c,\max(-c,g_i))$. Kayıbın **her** parametreye göre her kısmi türevi kesilir. $c$ ayarlanabilir bir hiperparametredir.
- **Norma göre:** $\|g\|>c$ ise tüm vektör $g\leftarrow g\cdot c/\|g\|$ ile ölçeklenir.

:::örnek Yön değişimi
$g=[0.9,\,100]$: çoğunlukla ikinci eksen yönünü gösterir. **Değere göre** $c=1$ ile $[0.9,\,1.0]$ olur — artık iki eksenin aşağı yukarı köşegenini gösteriyor: yön $\approx41.5°$ döndü. **Norma göre** $c=1$ ile $\|g\|=100.004$, sonuç $[0.00899964,\,0.9999595]$: yön tam korunur, ama birinci bileşen neredeyse yok olur. Değere göre kırpma yönü değiştirse de pratikte çoğu zaman iyi çalışır; yönü korumak istiyorsan norma göre kırp.
:::

::lab l2

Eğitim sırasında gradyan normunu izle (TensorBoard ya da basit bir kayıt). Gradyanlar patlıyorsa hem değere hem norma göre kırpmayı farklı eşiklerle dene ve doğrulama setinde hangisinin daha iyi olduğuna bak.

:::ayrıntı PyTorch'ta kırpma
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # global norm, döndürdüğü değer kırpma ÖNCESİ norm
# ya da: torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
optimizer.step()
```
`clip_grad_norm_` normu **tüm parametrelerin birleştirilmiş vektörü** üzerinden hesaplar (global norm): her tensör aynı çarpanla ölçeklenir. Kırpma `backward()`'dan sonra, `step()`'ten önce yapılır. Karışık duyarlıkla (AMP) eğitiyorsan önce `scaler.unscale_(optimizer)` çağır; yoksa ölçeklenmiş gradyanı kırparsın. Döndürülen ön-kırpma normunu kaydetmek en ucuz teşhis aracıdır. x3 ekinde ikisini karşılaştırıyorsun.
:::

### Varsayılan yapılandırma ve istisnalar {#varsayilan}

Ders slaytlarındaki pratik kurallar (Géron'un önerileri) — başlangıç noktası, yasa değil:

| hiperparametre | varsayılan |
|---|---|
| ağırlık başlatma | He (M3.5) |
| aktivasyon | ReLU; derin ağ ve hesap bütçesi varsa Swish |
| normalizasyon | sığ ağda yok; derinse BatchNorm (M3.6) |
| düzenlileştirme | early stopping; gerekirse weight decay |
| optimizer | Nesterov momentum ya da AdamW (M3.4) |
| öğrenme oranı takvimi | performansa göre, 1cycle ya da kosinüs (M3.4) |

**Yalnız tam bağlı katmanlardan oluşan bir yığın** ise öz-normalize edebilir: LeCun başlatma + SELU, normalizasyon yok, gerekirse **alpha dropout**, Nesterov, aynı takvimler. Girdi özelliklerini standartlaştırmayı unutma.

**İstisnalar:**
- **Seyrek model** gerekiyorsa L1 kullan (ve eğitim sonunda çok küçük ağırlıkları sıfırla); daha da seyrek modeller için budama araçları (TensorFlow Model Optimization Toolkit, PyTorch'ta `torch.nn.utils.prune`). Bu, öz-normalizasyonu bozar — o durumda varsayılan yapılandırmaya dön.
- **Düşük gecikmeli model** (çok hızlı tahmin) gerekiyorsa: daha az katman, ReLU ya da Leaky ReLU gibi hızlı bir aktivasyon, eğitimden sonra BN katmanlarını önceki katmanlara katla (M3.6); seyrek bir model de yardım eder; son olarak sayı duyarlığını 32 bitten 16 hatta 8 bite düşür (nicemleme).

### Özet {#ozet}

| araç | ne yapar | ana hiperparametre | PyTorch |
|---|---|---|---|
| early stopping | doğrulama hatası artınca durur, en iyi ağırlıkları geri yükler | patience | elle döngü (d aşaması) |
| L2 / weight decay | ağırlıkları her adımda küçültür | $\lambda$ | `AdamW(weight_decay=…)`, `SGD(weight_decay=…)` |
| L1 | ağırlıkları sıfıra iter (seyreklik) | $\lambda$ | kayba `λ·Σ|w|` ekle |
| dropout | eğitimde nöronları $p$ olasılıkla düşürür, $\frac1{1-p}$ ile ölçekler | $p$ | `nn.Dropout(p)` |
| alpha dropout | SELU ağlarında ortalama/varyansı koruyan dropout | $p$ | `nn.AlphaDropout(p)` |
| MC dropout | çıkarımda $T$ dropout'lu tahminin ortalaması ve dağılımı | $T$ | yalnız dropout modüllerini `train()`'e al |
| max-norm | $\|w\|\le r$'ye izdüşüm | $r$ | `torch.renorm` |
| gradyan kırpma | gradyanın değerini ya da normunu sınırlar | $c$ | `clip_grad_norm_`, `clip_grad_value_` |

- Early stopping kuadratik kayıpta $\alpha\approx1/(\eta t)$ güçlü bir L2'ye denktir; ucuz ve hemen her zaman ilk tercihtir.
- Inverted dropout beklenen değeri korur ($\mathbb E[y]=x$) ve $\frac p{1-p}x^2$ varyanslı çarpımsal gürültü ekler; doğrusal modelde veriye uyarlanmış bir L2'dir; derin ağda $2^N$ alt ağın örtük topluluğudur.
- Adam'la L2 yerine AdamW kullan; bias ve normalizasyon parametrelerini weight decay'den muaf tut.
- Değere göre kırpma yönü değiştirir, norma göre kırpma korur; PyTorch'un `clip_grad_norm_`'u global normdur.
