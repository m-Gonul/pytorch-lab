### Problem: K sınıf, tek tutarlı olasılık {#problem}

Rakam tanıma (10 sınıf), görüntü sınıflandırma (1000 sınıf), dil modelinde sonraki kelime (50 000+ sınıf). M1.4'teki "bire karşı hepsi" $K$ ayrı ikili model eğitir; olasılıkları toplamı 1 olmayan, birbirinden habersiz sayılardır. İstediğimiz: her $x$ için $K$ sınıf üzerinde **tek bir olasılık dağılımı** ve onu en çok olabilirlikle eğitmek. M0.4'ün tarifi yine geçerli: model $y\mid x\sim\text{Kategorik}(p(x))$, kayıp NLL.

### Softmax {#softmax}

Her sınıf için bir logit: $z_k=w_k^\top x+b_k$, matris olarak $z=Wx+b\in\mathbb R^K$ (batch'te $Z=XW^\top+b$, $(n\times K)$). Logitleri olasılığa çevirmek için iki şart: pozitif olsunlar ve toplamları 1 olsun.
$$p_k=\mathrm{softmax}(z)_k=\frac{e^{z_k}}{\sum_{j=1}^Ke^{z_j}}.$$
Üs pozitifliği sağlar, bölme normalleştirir. Neden üs? (1) Kategorik dağılımın doğal parametrelemesi log-olasılıklardır: $\log p_k=z_k-\text{sabit}$ — M1.4'teki "log-odds doğrusal" varsayımının çok sınıflı karşılığı: $\log\frac{p_k}{p_j}=z_k-z_j$. (2) $K=2$'de sigmoid'e indirgenir: $\frac{e^{z_1}}{e^{z_0}+e^{z_1}}=\sigma(z_1-z_0)$.

**"Soft" max:** en büyük logit en büyük olasılığı alır, ama diğerleri de pay alır. Logitleri $T$ ile bölmek (**sıcaklık**): $T\to0$'da softmax argmax'a (one-hot), $T\to\infty$'da düzgün dağılıma yaklaşır. Bilgi damıtma ve dil modellerinde örnekleme bu düğmeyi kullanır.

:::tanım Kaydırma değişmezliği
Her $c$ için $\mathrm{softmax}(z+c\mathbf1)=\mathrm{softmax}(z)$: pay ve payda aynı $e^c$ ile çarpılır. Softmax yalnız logit **farklarına** bakar. İki sonucu: (1) parametreler tek değildir — tüm $w_k$'lara aynı vektörü eklemek modeli değiştirmez (L2 cezası ya da bir sınıfın logitini 0'a sabitlemek bu belirsizliği giderir); (2) sayısal kararlılık hilesi buna dayanır (aşağıda).
:::

Sıcaklığı ve kaydırmayı aşağıda dene.

::lab l1

### Kayıp: çapraz entropi ve log-sum-exp {#kayip}

Kategorik NLL, doğru sınıfın olasılığının negatif logudur:
$$\ell(z,y)=-\log p_y=-z_y+\log\sum_je^{z_j}=\mathrm{LSE}(z)-z_y.$$
$\mathrm{LSE}(z)=\log\sum_je^{z_j}$ **log-sum-exp**'tir: max'ın pürüzsüz bir yaklaşımı ($\max z\le\mathrm{LSE}(z)\le\max z+\log K$). Kayıp böylece "en büyük logit ile doğru sınıfın logiti arasındaki (pürüzsüz) fark"tır: doğru sınıf açık ara öndeyse ≈ 0, başka bir sınıf öndeyse aradaki fark kadar.

**Başlangıç değeri:** tüm logitler eşitken $p_k=1/K$ ve $\ell=\log K$. 10 sınıfta $\log10\approx2.30$. Eğitime başlarken kayıp bundan çok farklıysa (ör. 30) başlatmada bir sorun vardır — basit ama çok kullanışlı bir kontrol (Karpathy, "A recipe for training neural networks").

:::türetim Gradyan: ∂ℓ/∂z = p − onehot(y)
$\frac{\partial\,\mathrm{LSE}}{\partial z_k}=\frac{e^{z_k}}{\sum_je^{z_j}}=p_k$ (LSE'nin gradyanı softmax'tır) ve $\frac{\partial z_y}{\partial z_k}=[k=y]$. Böylece
$$\frac{\partial\ell}{\partial z_k}=p_k-[k=y],\qquad \frac{\partial\ell}{\partial z}=p-e_y.$$
Batch için: $P$ ve $Y$ $(n\times K)$ olasılık ve one-hot matrisleri, $G=\frac1n(P-Y)$, $\nabla_WL=G^\top X$ ($K\times d$), $\nabla_bL=\mathbf1^\top G$. (Shape kontrolü M0.2: $dW$, $W$ ile aynı shape.)
**Yorum:** her örnek doğru sınıfın logitini $1-p_y$ kadar yukarı, her yanlış sınıfınkini **o sınıfa verdiği olasılık** kadar aşağı iter. Gradyanın bileşenlerinin toplamı $\sum_k(p_k-[k=y])=1-1=0$: kaydırma değişmezliğinin türevdeki izi.
:::

Softmax'ın Jacobian'ı $\mathrm{diag}(p)-pp^\top$'dir (M0.3'te türettik) ve tam doludur; ama bunu çapraz entropinin gradyanıyla çarpınca ($\frac{\partial\ell}{\partial p}=-e_y/p_y$) sonuç yine $p-e_y$ çıkar. Bu yüzden kütüphaneler softmax ile çapraz entropiyi **tek bir işlem** olarak uygular: hem kararlı hem ucuz.

### Sayısal kararlılık: log-sum-exp hilesi {#lse}

Naif hesap $e^{z_j}$'yi doğrudan alır: $z=[1000,999,998]$ için $e^{1000}$ float64'te taşar (sınır ~709; float32'de ~88), sonuç `inf/inf = nan`. Tersine çok negatif logitlerde tüm $e^{z_j}$ sıfıra yuvarlanır, $\log0=-\infty$.

:::türetim LSE(z) = m + log Σ exp(zⱼ − m)
$m=\max_jz_j$ olsun. $\sum_je^{z_j}=e^m\sum_je^{z_j-m}$; log alınca $\mathrm{LSE}(z)=m+\log\sum_je^{z_j-m}$. Şimdi en büyük üs $e^0=1$: taşma imkânsız. Toplamda en az bir terim 1 olduğu için toplam $\ge1$, log'u $\ge0$: sıfıra yuvarlanma sorunu da yok. Softmax için aynı hile: $p_k=e^{z_k-m}/\sum_je^{z_j-m}$ (kaydırma değişmezliği). **Log-olasılıklar** gerekiyorsa $\log p_k=z_k-\mathrm{LSE}(z)$ ile doğrudan hesapla (`log_softmax`); önce $p$ hesaplayıp log almak küçük olasılıklarda yine $\log0$ üretir.
:::

PyTorch karşılıkları: `torch.logsumexp(z, dim=-1)`, `F.log_softmax(z, dim=-1)`, `F.cross_entropy(logits, y)` (= `log_softmax` + `nll_loss`, logit alır, hedef **sınıf indisleri** `(n,)` `long`). Model son katmanda **softmax uygulamaz**.

| yaygın hata | sonuç |
|---|---|
| `F.cross_entropy(F.softmax(z), y)` | softmax iki kez uygulanır: olasılıklar düzleşir, gradyanlar küçülür, sessizce kötü eğitim |
| hedef shape'i `(n, 1)` | indis hedefi `(n,)` olmalı; `y.squeeze(1)` ya da `y.view(-1)`. (One-hot float `(n, K)` hedef ise PyTorch ≥ 1.10'da olasılık hedefi olarak kabul edilir ve aynı kaybı verir.) |
| `dim` yanlış | `softmax(z, dim=0)` batch üzerinde normalleştirir: her sütun 1'e toplanır, satırlar değil |
| hedef `float` | `nll_loss` `long` indis bekler; hata verir |

### Geometri: karar bölgeleri {#geometri}

$x$, en büyük logitli sınıfa atanır. $k$ ile $j$ arasındaki sınır $z_k=z_j$, yani $(w_k-w_j)^\top x+(b_k-b_j)=0$: bir hiperdüzlem. Her sınıfın bölgesi $K-1$ yarım uzayın kesişimidir: **dışbükey bir çokyüzlü**. Softmax regresyonu bu yüzden doğrusal bir sınıflandırıcıdır; doğrusal olmayan sınırlar için öznitelikleri dönüştürmek ya da öğrenmek gerekir — sinir ağının yaptığı tam olarak budur: son katman yine bir softmax regresyonudur, önceki katmanlar ona doğrusal ayrılabilir öznitelikler hazırlar.

::lab l2

### Etiket yumuşatma ve diğer varyantlar {#varyantlar}

- **Etiket yumuşatma** (label smoothing, Szegedy ve ark. 2016): hedefi one-hot yerine $(1-\varepsilon)e_y+\varepsilon/K$ yap. Model hiçbir zaman %100 emin olmaya zorlanmaz; logitlerin sınırsız büyümesini (M1.4'teki ayrılabilir veri sorunu) engeller, kalibrasyonu çoğu zaman iyileştirir. `F.cross_entropy(..., label_smoothing=0.1)`.
- **Sınıf ağırlıkları:** dengesiz verilerde nadir sınıfların kaybını büyüt (`weight=`).
- **Çok etiketli** sınıflandırma (bir görüntüde hem kedi hem köpek): softmax yanlış araçtır (sınıflar yarışır); her sınıf için bağımsız sigmoid + BCE kullanılır (M1.4).
- **Çok büyük K** (dil modelleri): softmax'ın paydası tüm sözlük üzerinden toplam ister; bellek ve hesap darboğazıdır. Örneklenmiş softmax, hiyerarşik softmax ve "kesilmiş çapraz entropi" gibi teknikler bunun için vardır (M4.12).

### Değerlendirme {#degerlendirme}

- **Doğruluk** ve **top-k doğruluk** (doğru sınıf ilk $k$ tahmin içinde mi; ImageNet'te top-5).
- **Karışıklık matrisi** $C_{ij}$ = gerçek sınıfı $i$ olup $j$ tahmin edilen örnek sayısı: hangi sınıfların birbirine karıştığını gösterir (modülün **e** aşamasında döngüsüz, `bincount` ile kuracaksın).
- **Ortalama çapraz entropi** (ve üssü **perplexity** $e^{\bar\ell}$: "model ortalamada kaç sınıf arasında kararsız", dil modellerinde standart).

### Özet {#ozet}

| | ikili (M1.4) | çok sınıflı (M1.5) |
|---|---|---|
| model | $p=\sigma(w^\top x+b)$ | $p=\mathrm{softmax}(Wx+b)$ |
| kayıp | $\mathrm{softplus}(z)-yz$ | $\mathrm{LSE}(z)-z_y$ |
| gradyan $\partial\ell/\partial z$ | $p-y$ | $p-e_y$ |
| kararlılık | softplus | log-sum-exp ($m=\max z$ çıkar) |
| PyTorch | `BCEWithLogitsLoss` | `F.cross_entropy` (logit + indis) |
| başlangıç kaybı | $\log2$ | $\log K$ |

**Akılda kalsın:** softmax logit farklarına bakar; çapraz entropi = LSE − doğru logit; gradyan $p-$ one-hot. Her zaman logit üzerinden, log-sum-exp ile hesapla; softmax'ı modelin içine koyma. Başlangıç kaybının $\log K$ olduğunu kontrol et.

**Kaynaklar:** CS231n, "Linear classification: Softmax classifier" ve Assignment 1 (vektörize softmax kaybı); d2l.ai §4.1; Bishop, *Pattern Recognition and Machine Learning* §4.3.4; Goodfellow ve ark., *Deep Learning* §6.2.2.3 ve §4.1 (taşma ve log-sum-exp); Szegedy ve ark. (2016), "Rethinking the Inception architecture" (etiket yumuşatma); Karpathy (2019), "A recipe for training neural networks"; PyTorch `F.cross_entropy`, `torch.logsumexp` belgeleri.
