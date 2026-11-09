### Problem: düz SGD neden yetmez? {#problem}

M1.2'de SGD'nin iki zaafını gördük: (1) **kötü koşullu** kayıp yüzeylerinde (dar vadiler) en dik yön kararlılığı, en yatık yön hızı belirler; ilerleme $\sim1/\kappa$ hızındadır. (2) **Gürültü**: sabit öğrenme oranıyla optimumun etrafında bir gürültü topunda takılır. Derin ağlarda üçüncüsü eklenir: (3) farklı parametreler **çok farklı ölçekte** gradyan alır — bazı katmanlar, bazı gömme satırları (nadir kelimeler) çok seyrek ve küçük gradyan görür. Tek bir küresel öğrenme oranı herkese uymaz.

Bu modüldeki optimizer'lar bu üç soruna üç fikirle cevap verir: **momentum** (geçmiş gradyanları biriktir: 1 ve kısmen 2), **uyarlamalı ölçekleme** (her parametreyi kendi gradyan büyüklüğüne göre ölçekle: 3, kısmen 1), **öğrenme oranı takvimi** (zamanla adımı küçült: 2). Adam ilk ikisini birleştirir; pratikte üçü birlikte kullanılır.

Hepsinin iskeleti aynıdır: gradyandan bir **durum** güncelle, durumdan bir **adım** hesapla. PyTorch'un `torch.optim.Optimizer`'ı her parametre için bir `state` sözlüğü tutar (momentum tamponu, moment kestirimleri, adım sayacı) — modülün **d** aşamasında kendi optimizer'ını bu yapıyla yazacaksın.

### Momentum ve Nesterov {#momentum}

M1.2'de türettik; kısa tekrar ve ders slaytlarındaki biçimiyle:

:::tanım Momentum (Polyak 1964)
Keras/ders biçimi: $m\leftarrow\beta m-\eta\nabla_\theta C$, $\theta\leftarrow\theta+m$. PyTorch biçimi: $u\leftarrow\beta u+g$, $\theta\leftarrow\theta-\eta u$. İkisi aynı yörüngeyi verir ($m=-\eta u$). $\beta$ "sürtünme"yi ayarlar: 0 = tam sürtünme (düz SGD), 1'e yakın = sürtünmesiz. Sabit bir gradyan altında hız bir **limit hıza** ulaşır: $u\to g/(1-\beta)$ — etkin adım $\eta/(1-\beta)$, $\beta=0.9$'da 10 kat. Yerel minimumlardan ve platolardan "yuvarlanarak" çıkmaya da yardım eder.
:::

**Nesterov hızlandırılmış gradyanı (NAG):** gradyanı mevcut konumda değil, momentumun götüreceği **ileri bakış** noktasında ölç: $m\leftarrow\beta m-\eta\nabla C(\theta+\beta m)$, $\theta\leftarrow\theta+m$. Momentum genelde doğru yönü gösterdiği için ileri noktadaki gradyan biraz daha doğrudur; vadinin dibine yaklaşırken salınım erken frenlenir. Pratikte düz momentumdan neredeyse her zaman biraz iyidir (`SGD(nesterov=True)`).

### AdaGrad: parametre başına öğrenme oranı {#adagrad}

:::tanım AdaGrad (Duchi, Hazan, Singer 2011)
$$s\leftarrow s+g\otimes g,\qquad \theta\leftarrow\theta-\eta\,g\oslash\sqrt{s+\varepsilon}.$$
$\otimes$, $\oslash$ eleman eleman çarpma ve bölme; $\varepsilon\approx10^{-10}$ sıfıra bölmeyi önler. Her koordinat, **geçmişteki tüm** gradyan karelerinin toplamının kareköküyle ölçeklenir.
:::

**Etkisi:** dik yönlerde (büyük gradyanlar) $s$ hızla büyür, adım küçülür; yatık yönlerde adım görece büyük kalır — güncellemeler optimuma daha doğrudan yönelir, ve $\eta$ ayarına daha az duyarlıdır. Seyrek özniteliklerde (nadir kelimeler) o koordinat nadiren güncellendiği için $s$ küçük kalır ve her güncelleme büyük olur: bu yüzden AdaGrad seyrek problemlerde (metin, öneri) parlar.

**Kusuru:** $s$ hiç küçülmez, sürekli büyür; etkin öğrenme oranı $\eta/\sqrt{s}$ sıfıra gider. Basit kuadratiklerde iyidir, ama derin ağ eğitiminde **çok erken durur** — optimuma ulaşmadan adımlar sönmüş olur.

### RMSProp: unutmayı ekle {#rmsprop}

:::tanım RMSProp (Hinton, Coursera ders notu 2012)
$$s\leftarrow\rho s+(1-\rho)\,g\otimes g,\qquad \theta\leftarrow\theta-\eta\,g\oslash\sqrt{s+\varepsilon}.$$
AdaGrad'ın toplamı yerine **üstel hareketli ortalama** ($\rho\approx0.9$): yalnız son ~$1/(1-\rho)$ adımın gradyan büyüklüğü hatırlanır, etkin öğrenme oranı sıfıra sönmez. (PyTorch'ta $\rho$'nun adı `alpha`.)
:::

Ders alıştırması 3'teki **unutma faktörü** tam olarak bu $\rho$'dur: EMA'nın ağırlıkları $(1-\rho)\rho^k$, toplamı 1; etkin pencere $\approx1/(1-\rho)$. Aşağıdaki deneyde EMA'nın gecikme–gürültü takasını ve sıfırdan başlamanın yarattığı yanlılığı gör.

::lab l2

### Adam: momentum + RMSProp + yanlılık düzeltmesi {#adam}

:::tanım Adam (Kingma ve Ba 2015)
$$m\leftarrow\beta_1m+(1-\beta_1)g\qquad v\leftarrow\beta_2v+(1-\beta_2)g\otimes g$$
$$\hat m=\frac m{1-\beta_1^t}\qquad\hat v=\frac v{1-\beta_2^t}\qquad\theta\leftarrow\theta-\eta\,\hat m\oslash(\sqrt{\hat v}+\varepsilon)$$
Varsayılanlar: $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$, $\eta=10^{-3}$. $m$: gradyanın birinci moment kestirimi (momentum), $v$: ikinci moment (gradyan büyüklüğü, RMSProp).
:::

:::türetim Neden yanlılık düzeltmesi?
$m_0=0$ ile başlayınca $m_t=(1-\beta_1)\sum_{k=1}^t\beta_1^{t-k}g_k$. Gradyanların beklenen değeri sabit ($\mathbb E g_k=\mu$) ise $\mathbb E m_t=(1-\beta_1)\mu\sum_{k=0}^{t-1}\beta_1^k=(1-\beta_1^t)\mu$. Yani $m_t$, gerçek ortalamanın yalnız $(1-\beta_1^t)$ kesrini kestirir: ilk adımlarda sıfıra doğru **yanlıdır**. $1-\beta_1^t$'ye bölmek bunu düzeltir. $\beta_2=0.999$ için yanlılık çok daha uzun sürer: $t=10$'da $1-0.999^{10}\approx0.01$ — düzeltme olmasa $v$ ~100 kat küçük, $\sqrt v$ ~10 kat küçük olurdu ve ilk adımlar ~10 kat **büyük** atılırdı (eğitimin en kırılgan anında).
**Ne söylemez:** düzeltme yalnız sıfırdan başlamanın yanlılığını giderir; gradyanların dağılımı değişiyorsa (genelde öyle) EMA yine gecikmeli bir kestirimdir.
:::

**Adam'ın adımı neden ölçekten bağımsız?** Gradyanı 100 ile çarpsan $\hat m$ ve $\sqrt{\hat v}$ ikisi de 100 kat büyür, oranları değişmez: adım büyüklüğü kabaca $\eta$ ile sınırlıdır (her koordinatta $|\Delta\theta|\lesssim\eta$). Bu, $\eta$'yı "parametre uzayında adım boyu" olarak yorumlanabilir yapar ve kaybın ölçeğine karşı dayanıklılık sağlar.

**Akrabalar:**

| yöntem | fark | not |
|---|---|---|
| AdaMax | $\sqrt v$ yerine $L_\infty$: $s\leftarrow\max(\beta_2s,|g|)$ | aşırı büyük tekil gradyanlara daha dayanıklı; pratikte nadiren daha iyi |
| Nadam | Adam + Nesterov ileri bakışı | genelde Adam'dan biraz hızlı yakınsar |
| AdamW | ağırlık sönümü gradyandan ayrık (aşağıda) | modern varsayılan |
| Lion, Sophia, Shampoo, Muon | işaret tabanlı / ikinci dereceden / matris ön-koşullu | büyük modellerde araştırma sınırı |

Aşağıdaki deneyde aynı kötü koşullu vadide SGD, momentum, Nesterov, RMSProp ve Adam'ı karşılaştır; vadiyi döndürünce koordinat başına ölçeklemenin avantajının nasıl kaybolduğuna dikkat et.

::lab l1

:::yanılgı "Adam her şeyi koşul sorunundan kurtarır"
Adam'ın ölçeklemesi **köşegendir**: her koordinatı ayrı ölçekler. Kayıp vadisi eksenlere hizalıysa harikadır; vadi eksenlere göre eğikse (koordinatlar arası güçlü ilişki) köşegen bir ön-koşullayıcı eğriliği düzeltemez (deneyde açıyı 45°'ye getir). Tam ön-koşullama (Newton, K-FAC, Shampoo) bu ilişkileri de hesaba katar ama çok daha pahalıdır.
:::

### Weight decay: Adam'da L2 ≠ sönüm (AdamW) {#adamw}

M1.3'te düz SGD için L2 cezasının "her adımda ağırlıkları $(1-\eta\lambda)$ ile çarpmak" olduğunu gördük. Adam'da bu eşdeğerlik **bozulur**:

- **Adam + L2** (`Adam(weight_decay=λ)`): $\lambda\theta$ gradyana eklenir, sonra **ölçeklenir**: $\Delta\theta\propto(g+\lambda\theta)/\sqrt{\hat v}$. Büyük gradyan geçmişi olan parametreler ($\hat v$ büyük) **daha az** sönümlenir — tam da en çok güncellenen, düzenlileştirilmesi en çok gereken parametreler.
- **AdamW** (Loshchilov ve Hutter 2019): sönümü ölçeklemenin dışında uygula: $\theta\leftarrow\theta-\eta\lambda\theta-\eta\,\hat m/(\sqrt{\hat v}+\varepsilon)$. Her parametre aynı oranda sönümlenir; L2'nin SGD'deki anlamı geri gelir. Genelleme belirgin iyileşir; Transformer eğitiminde standarttır. (PyTorch'ta sönüm $\eta$ ile çarpılır: `lr` değişince etkin sönüm de değişir.)

### Uyarlamalı yöntemler ve genelleme {#genelleme}

Adam genelde **daha hızlı** yakınsar ve ayarlaması kolaydır. Ama bazı problemlerde (özellikle görüntü sınıflandırmada) iyi ayarlanmış SGD + momentum **daha iyi genelleyen** çözümler bulur (Wilson ve ark. 2017). Pratik öneriler:

- Önce Adam/AdamW ile hızlı bir taban çizgisi kur; varsayılan $\eta=10^{-3}$ (Transformer'larda $10^{-4}$–$10^{-3}$, ısınmayla).
- Doğrulama hatası beklenenden kötüyse bir de SGD + Nesterov dene (ve öğrenme oranı takvimini ciddi ayarla). "Veri kümen uyarlamalı gradyanlara alerjik olabilir."
- Transformer, dil modelleri, GAN, RL: Adam/AdamW neredeyse her zaman. ResNet/ImageNet tipi CNN'ler: SGD + momentum hâlâ yaygın.

### Öğrenme oranı takvimleri {#takvim}

Büyük $\eta$ hızlı ilerler ama optimumun etrafında zıplar; küçük $\eta$ sonunda iyi bir noktaya oturur ama çok yavaş. Büyük başla, zamanla küçült:

| takvim | formül | not |
|---|---|---|
| power | $\eta_t=\eta_0/(1+t/s)^c$ | önce hızlı, sonra yavaş düşer; $c$ genelde 1 |
| üstel | $\eta_t=\eta_0\cdot0.1^{t/s}$ | her $s$ adımda 10'da birine |
| parçalı sabit | belirli epoch'larda ×0.1 | klasik ResNet takvimi; ayarı elle |
| performansa göre | doğrulama kaybı $N$ epoch iyileşmezse ×$\lambda$ | `ReduceLROnPlateau`; ek ayar az |
| kosinüs (+ ısınma) | $\eta_t=\tfrac{\eta_0}2(1+\cos\frac{\pi t}T)$ | pürüzsüz; modern varsayılan; başta doğrusal ısınma |
| 1cycle | ilk yarı $\eta_0/10\to\eta_0$, ikinci yarı geri, son birkaç epoch çok küçük | Smith (2018); çoğu zaman hızlı ve iyi |

**Isınma (warmup)** neden? Eğitimin başında parametreler rastgeledir, kayıp yüzeyi keskin olabilir ve Adam'ın $\hat v$ kestirimleri henüz güvenilmezdir; büyük bir ilk adım eğitimi geri dönülmez biçimde bozabilir. İlk birkaç yüz–bin adımda $\eta$'yı sıfırdan doğrusal artırmak bunu önler (özellikle Transformer'larda ve büyük batch'lerde).

::lab l3

### Uygulama: torch.optim {#pytorch}

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=E * len(loader))
for x, y in loader:
    loss = loss_fn(model(x), y)
    opt.zero_grad(set_to_none=True)   # .grad'ları None yap (bellek ve hız)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # M3.7
    opt.step(); sched.step()          # takvim her adımda (ya da her epoch'ta, takvime göre)
```

- **Parametre grupları:** farklı katmanlara farklı `lr` ya da `weight_decay` (M1.3: bias ve normalizasyon parametrelerine sönüm yok; ince ayarda alt katmanlara küçük lr).
- **Durum belleği:** Adam her parametre için iki tensör daha tutar ($m$, $v$): optimizer belleği = parametre belleğinin 2 katı (7B'lik bir modelde fp32'de 56 GB, M0.5).
- **`_foreach` ve `fused`:** çok sayıda küçük parametre tensörünü tek tek güncellemek birçok küçük kernel demektir; `torch._foreach_*` işlemleri ve `fused=True` seçeneği bunları birleştirir (modülün **e** aşaması).
- **`state_dict`:** kontrol noktası (checkpoint) alırken optimizer durumunu da kaydet; yoksa eğitime devam ederken Adam'ın momentleri sıfırdan başlar.

### Özet {#ozet}

| yöntem | durum | adım | ne zaman |
|---|---|---|---|
| SGD + momentum | $u\leftarrow\beta u+g$ | $-\eta u$ | CNN'ler, iyi ayarlı takvimle güçlü genelleme |
| Nesterov | ileri bakış gradyanı | | momentumdan biraz iyi |
| AdaGrad | $s\mathrel{+}=g^2$ | $-\eta g/\sqrt s$ | seyrek problemler; derin ağda erken söner |
| RMSProp | $s\leftarrow\rho s+(1-\rho)g^2$ | $-\eta g/\sqrt s$ | RNN'ler, RL |
| Adam | $m$, $v$ + yanlılık düzeltmesi | $-\eta\hat m/\sqrt{\hat v}$ | varsayılan; hızlı ve dayanıklı |
| AdamW | Adam + ayrık sönüm | $-\eta\lambda\theta$ ayrıca | Transformer'lar, modern varsayılan |

**Kaynaklar:** Toygar, BİL 442 Bölüm 11 slaytları (momentum, Nesterov, AdaGrad, RMSProp, Adam, AdaMax, Nadam, AdamW, genelleme notu, öğrenme oranı takvimleri) ve Alıştırma 3; Géron, *Hands-On Machine Learning* (3. bs.) Bölüm 11; Kingma ve Ba (2015), Adam; Loshchilov ve Hutter (2019), AdamW; Duchi, Hazan, Singer (2011), AdaGrad; Sutskever ve ark. (2013), momentum ve başlatma; Dozat (2016), Nadam; Wilson ve ark. (2017), "The marginal value of adaptive gradient methods"; Smith (2018), 1cycle; Loshchilov ve Hutter (2017), SGDR (kosinüs takvimi); d2l.ai §12.8–12.11; PyTorch `torch.optim` belgeleri.
