### Problem: tam gradyan pahalı {#problem}

M1.1'de gradyan inişi her adımda **tüm** veriyi kullandı: $g=\nabla L(\theta)=\frac1n\sum_{i=1}^n\nabla\ell_i(\theta)$, $\ell_i$ tek örneğin kaybı. ImageNet'te $n\approx1.3$ milyon; tek bir adım için bir ResNet'i 1.3 milyon görüntüde ileri ve geri çalıştırmak saatler sürer. Oysa iyi bir model binlerce adım ister. Bu modül iki soruyu yanıtlar:

1. **Gradyanı ucuza tahmin etmek:** tüm veri yerine küçük bir rastgele alt küme (mini-batch) yeter mi? Gürültünün bedeli nedir?
2. **Daha akıllı adım:** M1.1'deki koşul sayısı sorununu (dar vadide zikzak) adım kuralını değiştirerek hafifletebilir miyiz? (Momentum.)

Önce tek boyutta GD'nin davranışını tamamen anlayalım; her şey buna indirgenir.

### Tek boyutta GD: her şey 1 − ηa'da {#tek-boyut}

$f(x)=\frac a2x^2$ ($a>0$ eğrilik) için $f'(x)=ax$ ve
$$x_{t+1}=x_t-\eta ax_t=(1-\eta a)\,x_t\quad\Rightarrow\quad x_t=(1-\eta a)^t\,x_0.$$
Geometrik bir dizi. Dört rejim:

| $\eta a$ | $1-\eta a$ | davranış |
|---|---|---|
| $0<\eta a<1$ | $(0,1)$ | tekdüze yakınsama |
| $\eta a=1$ | $0$ | tek adımda minimum (Newton adımı) |
| $1<\eta a<2$ | $(-1,0)$ | işaret değiştirerek (salınarak) yakınsama |
| $\eta a>2$ | $<-1$ | ıraksama |

M1.1'de gördüğümüz gibi çok boyutlu kuadratik, Hessian'ın özvektörleri yönünde bağımsız 1-B problemlere ayrılır; her yönün kendi $a=\lambda_k$'sı vardır. Tek bir $\eta$ hepsine birden uymak zorundadır: kararlılık için $\eta<2/\lambda_{\max}$, bu yüzden en yatık yön ($\lambda_{\min}$) $\eta\lambda_{\min}\le2/\kappa$ hızında ilerler. Önce aşağıda 1-B'de sınırları bul.

Aşağıdaki sürüklenebilir şekilde başlangıç noktasını elle seç ve adımları tek tek at: her adımın yatay uzaklığı $-\eta f'(x_t)$, dikey iniş eğriye. $\eta a<1$ iken merdiven içe sarar; $1<\eta a<2$ iken her iniş karşı yamaça geçer; $\eta a>2$ iken dışa.

::lab l6

::lab l1

**Gerçek kayıplar kuadratik değildir**, ama minimumun yakınında Taylor açılımıyla yaklaşık kuadratiktir (Hessian = yerel eğrilik). Bu yüzden "öğrenme oranı yerel eğriliğe göre fazla büyükse eğitim patlar" sezgisi derin ağlarda da geçerlidir; sadece eğrilik eğitim boyunca değişir (öğrenme oranı ısınması (warmup) ve takvimlerin bir nedeni, M3.4).

### Stokastik gradyan: yansız ama gürültülü {#sgd}

Rastgele bir örnek $i$ seç (eşit olasılıkla). Tek örneğin gradyanı $\nabla\ell_i(\theta)$:
$$\mathbb E_i[\nabla\ell_i(\theta)]=\frac1n\sum_i\nabla\ell_i(\theta)=\nabla L(\theta).$$
**Yansız** bir tahmin! Ama gürültülüdür: varyansı, örnek gradyanlarının birbirinden ne kadar farklı olduğuna bağlıdır. $B$ örneklik bir mini-batch'in ortalaması
$$g_B=\frac1B\sum_{i\in\mathcal B}\nabla\ell_i,\qquad \mathbb E[g_B]=\nabla L,\qquad \mathrm{Cov}[g_B]\approx\frac{\Sigma}{B},$$
$\Sigma$ tek örnek gradyanlarının kovaryansı (M0.4: ortalamanın varyansı $\sigma^2/n$). **Gürültünün standart sapması $1/\sqrt B$ ile azalır**, maliyet ise $B$ ile artar.

| yöntem | adımda | adım maliyeti | gürültü std | epoch başına adım |
|---|---|---|---|---|
| tam batch GD | $n$ örnek | $O(n)$ | 0 | 1 |
| SGD | 1 örnek | $O(1)$ | $\propto1$ | $n$ |
| mini-batch | $B$ örnek | $O(B)$ | $\propto1/\sqrt B$ | $\lceil n/B\rceil$ |

:::tanım Epoch ve karıştırma
Bir **epoch**, eğitim verisinin tamamının bir kez kullanılmasıdır. Pratikte her epoch başında indisler **rastgele karıştırılır** ve ardışık, örtüşmeyen $B$'lik parçalara bölünür (yerine koymadan örnekleme). Bu, her örneği epoch başına tam bir kez kullanır; yerine koyarak örneklemeye göre biraz daha az varyanslıdır. **Karıştırmamak** tehlikelidir: veri sınıfa göre sıralıysa model önce bir sınıfı, sonra diğerini "öğrenir" ve döngüye girer. Aşağıdaki deneyde "x'e göre sıralı" seçeneği bu tehlikeyi canlı gösterir: her batch tek $x$ bölgesinden geldiği için batch gradyanı yansızlığını yitirir.
:::

**Neden işe yarar?** Gürültülü ama ucuz adımlar, aynı hesap bütçesinde (aynı epoch sayısında) çok daha fazla adım demektir. Optimumdan uzaktayken gradyan büyük, gürültü görece küçüktür: her adım doğru yöne gider. Optimuma yaklaştıkça gradyan küçülür, gürültü baskın hâle gelir.

:::türetim Gürültü topu: sabit η'da SGD neden durmaz?
1-B kuadratikte $f(x)=\frac a2x^2$, gürültülü gradyan $ax_t+\xi_t$, $\mathbb E\xi=0$, $\mathrm{Var}\,\xi=s^2/B$. $x_{t+1}=(1-\eta a)x_t-\eta\xi_t$. Varyans: $V_{t+1}=(1-\eta a)^2V_t+\eta^2s^2/B$. Durağan değer ($V_{t+1}=V_t=V$):
$$V=\frac{\eta^2s^2/B}{1-(1-\eta a)^2}=\frac{\eta s^2}{B\,a\,(2-\eta a)}\approx\frac{\eta\,s^2}{2aB}\quad(\eta a\ll1).$$
Sabit öğrenme oranıyla SGD optimuma **yakınsamaz**; etrafında std'si $\sqrt V\propto\sqrt{\eta/B}$ olan bir bulut içinde dolaşır. Topu küçültmenin yolları: $\eta$'yı azaltmak (öğrenme oranı takvimi), $B$'yi artırmak. $\eta/B$ oranının önemli olması, "batch'i $k$ katına çıkarırsan öğrenme oranını da $k$ katına çıkar" (doğrusal ölçekleme kuralı, Goyal ve ark. 2017) sezgisinin kaynağıdır — kural yalnız belli bir $B$'ye kadar geçerlidir.
:::

**Azalan öğrenme oranı.** Robbins–Monro (1951) koşulları: $\sum_t\eta_t=\infty$ (yeterince yol katedilebilsin) ve $\sum_t\eta_t^2<\infty$ (gürültü birikimi sonlu olsun). $\eta_t\propto1/t$ ikisini de sağlar. Derin öğrenmede pratikte basamaklı düşürme, kosinüs takvimi ya da ters karekök kullanılır (M3.4).

| takvim | $\sum\eta_t=\infty$ | $\sum\eta_t^2<\infty$ | gürültü topu |
|---|---|---|---|
| sabit $\eta$ | ✓ | ✗ | küçülmez: optimum çevresinde sabit std |
| $\eta_t=c/t$ | ✓ | ✓ | yakınsar; ilk adımlar için $c$ kritik |
| $\eta_t=c/\sqrt t$ | ✓ | ✗ | yavaş küçülür; tam yakınsamaz |
| $\eta_t=c/t^{\alpha},\ \alpha\in(\frac12,1)$ | ✓ | ✓ | pratikte güzel denge |
| basamaklı (her $k$ epoch'ta ×0.1) | ✓ | ✓ | basamakta topun ani daralması |

:::ayrıntı Polyak–Ruppert ortalaması: yörüngenin ortalaması
SGD yörüngesi gürültü topunun içinde gezer; ama **iteratların ortalaması** çok daha sakin bir yol izler: $\bar\theta_t=\frac1t\sum_{s\le t}\theta_s$. Polyak ve Juditsky (1992): $\eta_t\propto t^{-\alpha}$, $\alpha\in(\frac12,1)$ için $\bar\theta_t$ **c sabitine duyarsız** biçimde $O(1/t)$ hızda optimuma yaklaşır — en iyi $c$ ile koşan ham SGD'nin asimptotik hızına ulaşır, ama $c$'yi ayarlamaya gerek olmadan. Uygulamadaki akrabası üstel ağırlıklı ortalama (EMA): $\bar\theta_t=(1-\gamma)\bar\theta_{t-1}+\gamma\theta_t$. Sayısal deney bu modülün ödevinde.
:::

Aşağıda aynı epoch bütçesinde tam batch GD (mavi) ile mini-batch'i (turuncu) karşılaştır; $B$'yi 1'e indir, gürültü topunu gör; takvimi "azalan" yap.

::lab l2

:::yanılgı "Büyük batch her zaman daha iyidir, çünkü gradyan daha doğru"
Büyük batch adım başına daha doğrudur ama epoch başına daha az adım atar. $B$'yi iki katına çıkarmak gürültüyü yalnız $\sqrt2$ kat azaltır, hesabı 2 katına çıkarır. Belli bir "kritik batch boyu"nun üzerinde, aynı doğruluğa ulaşmak için gereken toplam hesap artmaya başlar (McCandlish ve ark. 2018). Ayrıca küçük batch'in gürültüsü bir tür düzenlileştirme gibi davranıp bazen genelleştirmeyi iyileştirir (Keskar ve ark. 2017, "keskin ve düz minimumlar" — tartışmalı bir sonuçtur). GPU'da çok küçük batch donanımı doyurmaz: adım başına süre $B$ ile orantılı düşmez. Seçim bu üç etkinin dengesidir.
:::

### Momentum: ağır top {#momentum}

Dar bir vadide GD, dik duvarlar arasında zikzak yaparken vadi boyunca yavaş ilerler (M1.1). Fikir: adımı yalnız şimdiki gradyana değil, **geçmiş gradyanların birikimine** göre at. Fiziksel benzetme: sürtünmeli bir yüzeyde yuvarlanan ağır top — tutarlı yönlerde hız kazanır, salınan yönlerde salınımlar birbirini götürür.

:::tanım Momentum (ağır top, Polyak 1964)
$$v_t=\beta v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,v_t,\qquad v_0=0.$$
$\beta\in[0,1)$ "sürtünme"nin tersidir (tipik 0.9). Açınca $v_t=g_t+\beta g_{t-1}+\beta^2g_{t-2}+\dots$: gradyanların üstel ağırlıklı toplamı; etkili bellek ~$1/(1-\beta)$ adım. Sabit bir gradyan altında $v\to g/(1-\beta)$: adım **$1/(1-\beta)$ kat** büyür ($\beta=0.9$'da 10 kat).

PyTorch `SGD(momentum=β)` tam olarak bu formülü kullanır (`dampening=0`). Bazı kaynaklar $v_t=\beta v_{t-1}+(1-\beta)g_t$ (EMA biçimi) yazar: bu, etkili öğrenme oranını $(1-\beta)$ ile çarpmakla eşdeğerdir — aynı $\eta$ ile iki biçim **farklı** davranır.

:::ayrıntı İki yazım biçimi: aynı (η, β) ile neden farklı davranır
Sabit bir $g$ altında ağır topun hızı $v\to g/(1-\beta)$'ya gider; EMA biçimininki $v\to g$'ye. Fark tam olarak etkin öğrenme oranıdır: EMA biçimindeki bütün hızlar $(1-\beta)$ ile ölçeklendiği için EMA + $\eta$, ağır top + $\eta(1-\beta)$ ile birebir aynıdır. Dolayısıyla **aynı (η, β) çifti iki biçimde aynı eğitim değildir**: EMA biçiminde β'yı büyütmek adımı büyütmez, yalnız geçmişin uzar ($1/(1-\beta)$ adımlık bellek); ağır topta β hem belleği uzatır hem adımı $1/(1-\beta)$ katına çıkarır. PyTorch'un `SGD(momentum=β)`'si ağır top biçimidir; Adam'ın birinci momenti (β₁) EMA biçimidir — yanlılık düzeltmesi $1-\beta^t$ çarpanı ilk adımlardaki küçüklüğü telafi eder (M3.4).
:::
:::

<figure class="fig wide"><svg viewBox="0 0 640 200" role="img" aria-label="Dar bir vadide GD zikzak yaparken momentum vadi boyunca ilerliyor"><g fill="none" class="mute"><ellipse cx="320" cy="100" rx="300" ry="34"/><ellipse cx="320" cy="100" rx="220" ry="25"/><ellipse cx="320" cy="100" rx="140" ry="16"/><ellipse cx="320" cy="100" rx="60" ry="7"/></g><circle cx="320" cy="100" r="3.5" class="ink-f"/><polyline points="40,70 70,128 98,74 124,124 148,78 170,120 190,82 208,117 225,85 240,114" class="cb w2" fill="none"/><text x="60" y="160" class="sm tb">GD: duvarlar arasında zikzak, vadi boyunca yavaş</text><polyline points="40,70 70,128 112,96 160,106 210,98 260,102 300,99 330,101" class="ca w2" fill="none"/><text x="300" y="40" class="sm ta">momentum: salınımlar söner, vadi boyunca hız birikir</text></svg><figcaption>Şekil 1 — Aynı başlangıç ve aynı η. Dik yöndeki gradyanlar her adımda işaret değiştirir, momentumda birbirini götürür; vadi boyunca gradyanlar hep aynı işaretlidir, birikir.</figcaption></figure>

:::türetim Momentum neden koşul sayısını hafifletir?
Kuadratikte bir özdeğer yönünde ($a=\lambda$) momentum iki değişkenli doğrusal bir sistemdir: $\begin{bmatrix}x_{t+1}\\v_{t+1}\end{bmatrix}$ önceki duruma bir $2\times2$ matrisle bağlıdır. Bu matrisin özdeğerlerinin büyüklüğü yakınsama hızını verir. Analiz (Polyak; ayrıntılı ve görsel anlatım: Goh, "Why Momentum Really Works") şunu gösterir: en iyi $\eta,\beta$ seçimiyle daralma oranı
$$\text{GD: }\frac{\kappa-1}{\kappa+1}\qquad\text{momentum: }\frac{\sqrt\kappa-1}{\sqrt\kappa+1}.$$
$\kappa=100$'de GD her adımda hatanın %2'sini, momentum %18'ini siler: aynı hassasiyete ~10 kat daha az adımla ulaşır. Etkili koşul sayısı $\kappa$'dan $\sqrt\kappa$'ya iner. Bu en iyi değerler $\beta^*=\big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^2$, $\eta^*=\big(\frac{2}{\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}}}\big)^2$'dir.
**Ne demez:** gürültülü gradyanlarda bu hızlanma büyük ölçüde kaybolur (momentum gürültüyü de biriktirir); dışbükey olmayan problemlerde garanti yoktur. $\beta$ çok büyükse sistem az sönümlenir ve minimumun etrafında uzun süre salınır.
:::

:::kanıt Ağır topta kararlılık bölgesi: yön başına 2×2 blok
Kuadratikte $H$'nin $\lambda$ özdeğerli bir öz yönünde durum $(x_t,v_t)$:
$$\begin{bmatrix}x_{t+1}\\ v_{t+1}\end{bmatrix}=\underbrace{\begin{bmatrix}1-\eta\lambda & -\eta\beta\\ \lambda & \beta\end{bmatrix}}_{M_\lambda}\begin{bmatrix}x_t\\ v_t\end{bmatrix},\qquad \det M_\lambda=\beta,\qquad \operatorname{tr}M_\lambda=1-\eta\lambda+\beta .$$
Özdeğerler $\mu_\pm=\frac{\operatorname{tr}\pm\sqrt{\operatorname{tr}^2-4\beta}}2$. Üç durum:

1. **Karmaşık kökler** ($\operatorname{tr}^2<4\beta$): $|\mu|=\sqrt\beta<1$ — her durumda sönümlü **spiral**; band $\eta\lambda\in\big((1-\sqrt\beta)^2,\ (1+\sqrt\beta)^2\big)$.
2. **Gerçek kökler, yakınsama**: Jüry koşulu $|\operatorname{tr}|<1+\det$ burada tam olarak $\eta\lambda<2(1+\beta)$'ya eşitlenir (öteki taraftaki $-\!1-\beta<\operatorname{tr}$ koşulu $\eta\lambda>0$'da otomatik sağlanır).
3. **Iraksama**: $\eta\lambda>2(1+\beta)$ — GD'nin $2$ sınırının $(1+\beta)$ katı: $\beta=0.9$'da $\eta$ 1.9 katına kadar çıkabilir.

Yakınsama bölgesi $\eta<2(1+\beta)/\lambda_{\max}$'tır; içinde $\eta\lambda_{\max}>(1-\sqrt\beta)^2$ iken dik yön spiralde, $\eta\lambda_{\min}<(1-\sqrt\beta)^2$ iken yatık yön gerçek-yavaş köklerde. **Mercek**, dik yönün spiral üst sınırı $\eta\lambda_{\max}=(1+\sqrt\beta)^2$ ile yatık yönün spiral alt sınırı $\eta\lambda_{\min}=(1-\sqrt\beta)^2$ eğrileri arasındaki bölgedir: orada her iki yön de tam $\sqrt\beta$ çarpanıyla sönümlenir. Uç, iki eğrinin kesişimidir: $\sqrt\lambda_{\max}(1-\sqrt\beta)=\sqrt\lambda_{\min}(1+\sqrt\beta)$ denklemden
$$\sqrt{\beta^*}=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\ \Longleftrightarrow\ \beta^*=\Big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\Big)^2,\qquad \eta^*=\frac{(1+\sqrt{\beta^*})^2}{\lambda_{\max}}=\Big(\frac{2}{\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}}}\Big)^2 .$$
Böylece önceki türetim kutusundaki en iyi çift ve $\big(\sqrt\kappa-1\big)/\big(\sqrt\kappa+1\big)$ daralma oranı, bu bölgenin geometrisinden gelir. **Ne söylemez:** gürültüde momentum gradyan gürültüsünün varyansını da $\propto1/(1-\beta^2)$ oranında biriktirir; dışbükey olmayan kayıplarda bu bölgeler yalnız yerel rehberdir.
:::

Aşağıdaki deney bu bölgeleri $(\eta,\beta)$ düzleminde boyar: ıraksama eğrisini, spiral bant sınırlarını ve işaretli mercek ucunu kendi elinle gez.

::lab l4

| $\kappa$ | GD çarpanı $\frac{\kappa-1}{\kappa+1}$ → $10^{-3}$ adım | momentum çarpanı $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$ → $10^{-3}$ adım |
|---|---|---|
| 10 | 0.818 → 35 | 0.520 → 11 |
| 100 | 0.980 → 346 | 0.818 → 35 |
| 1000 | 0.998 → 3454 | 0.939 → 110 |

**Nesterov hızlandırılmış gradyanı:** gradyanı mevcut noktada değil, momentumun götüreceği "ileri bakış" noktasında hesaplar: $v_t=\beta v_{t-1}+\nabla L(\theta_t-\eta\beta v_{t-1})$. Salınıma yaklaşırken frene daha erken basar; dışbükey problemlerde kanıtlanmış daha iyi hızı vardır (`SGD(nesterov=True)`).

:::türetim Nesterov neden erken frenler?
Dik öz yönde ağır top, bakış noktası $x-\eta\beta v$'de gradyan alır: $\lambda(x-\eta\beta v)$. Güncelleme
$$x_{t+1}=x_t-\eta\big(\beta v_t+\lambda x_t-\eta\beta\lambda v_t\big)=\big(1-\eta\lambda+\eta^2\beta\lambda^2\big)x_t-\eta\beta\big(1-\eta\lambda\big)v_t .$$
Dik yönde tepeyi aşan hız bileşeni $\eta\beta\lambda$ oranıyla küçülür: salınım başlamadan <b>hızı frenleyen ek terim</b> $1-\eta\beta\lambda$. Kuadratikte bu, η kararlılık sınırına dayanınca görülen uzun salınımları erken kırar; fark β büyük ve η sınıra yakınken belirginleşir. Genel düzgün dışbükey problemlerde Nesterov'un $O(1/k^2)$ hızı kanıtlıdır (Nesterov 1983); ağır top için böyle bir genel garanti yoktur (kuadratikte ikisi de $\sqrt\kappa$ hızına ulaşır).
:::

Aşama E'de 48'lik $(lr,\beta)$ ızgarasını tek tensörle eğitmiştin. Aşağıdaki harita aynı ızgaranın tarayıcı karşılığı: son kaybın $(\eta,\beta)$ düzlemindeki coğrafyası, bindirilmiş kararlılık sınırı ve teorik mercek ucu.

::lab l5

Aşağıdaki deneyde koşul sayısını, momentumu ve gradyan gürültüsünü değiştir: SGD (β = 0) ile ağır topu ve Nesterov'u aynı vadide karşılaştır.

::lab l3

### Pratikte eğitim döngüsü {#pratik}

```python
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                     drop_last=False, num_workers=2)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
for epoch in range(E):
    for xb, yb in loader:            # her epoch yeni bir karıştırma
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
```

- `shuffle=True` her epoch'ta yeni bir permütasyon üretir; tekrarlanabilirlik için `generator=torch.Generator().manual_seed(0)` ver.
- `drop_last=True` son küçük batch'i atar (BatchNorm'da tek örnekli batch'ten kaçınmak için, M3.6).
- Kayıp batch ortalamasıdır (`reduction="mean"`): öğrenme oranı $B$'den bağımsız kalır. `sum` kullanırsan etkili öğrenme oranı $B$ ile ölçeklenir.
- **Gradyan biriktirme:** bellek $B$'ye yetmiyorsa $k$ küçük batch'in gradyanlarını `zero_grad` çağırmadan topla, sonra tek `step` at; kaybı $k$'ya böl ki ortalama aynı kalsın.

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| 1-B GD | $x_{t+1}=(1-\eta a)x_t$; kararlılık $\eta<2/a$ |
| çok boyut | en dik yön kararlılığı, en yatık yön hızı belirler; hız ~$1/\kappa$ |
| mini-batch gradyanı | yansız; gürültü std $\propto1/\sqrt B$ |
| sabit η ile SGD | optimumu değil, std'si $\propto\sqrt{\eta/B}$ olan bir gürültü topunu bulur |
| takvim | $\sum\eta_t=\infty$, $\sum\eta_t^2<\infty$ |
| momentum | $v\leftarrow\beta v+g$, $\theta\leftarrow\theta-\eta v$; etkili adım $\eta/(1-\beta)$; hız ~$1/\sqrt\kappa$ |
| kararlılık bölgesi | ağır topta $\eta<2(1+\beta)/\lambda_{\max}$; mercek ucu $(\beta^*,\eta^*)$, çarpan $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ |

**Kaynaklar:** Stanford CS229 ders notları (toplu ve stokastik GD); d2l.ai §12.4–12.6; Goh (2017), "Why Momentum Really Works", Distill; Polyak (1964), ağır top yöntemi; Nesterov (1983), $O(1/k^2)$ hızlandırılmış gradyan; Polyak ve Juditsky (1992), stokastik yaklaşımda ortalama alma; Sutskever, Martens, Dahl ve Hinton (2013), momentum ve başlangıç değerinin önemi; Robbins ve Monro (1951), "A stochastic approximation method"; Bottou, Curtis ve Nocedal (2018), "Optimization methods for large-scale machine learning", SIAM Review (SGD kuramı, gürültü topu); Goyal ve ark. (2017), "Accurate, large minibatch SGD"; McCandlish ve ark. (2018), "An empirical model of large-batch training"; PyTorch `torch.optim.SGD` belgesi.
