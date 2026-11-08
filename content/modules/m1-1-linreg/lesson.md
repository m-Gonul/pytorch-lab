### Problem: girdiden sayısal bir çıktıyı tahmin etmek {#problem}

**Denetimli öğrenme**de elimizde $(x_i,y_i)$ çiftleri vardır: $x_i\in\mathbb R^d$ girdi (öznitelikler), $y_i\in\mathbb R$ hedef. Amaç, görmediğimiz yeni $x$'ler için $y$'yi iyi tahmin eden bir $f$ bulmaktır. Hedef sürekli bir sayıysa (fiyat, sıcaklık, süre) problem **regresyon**dur.

Bir öğrenme probleminin üç parçası vardır ve bu modülden itibaren her modelde aynı üçünü soracağız:

1. **Model (hipotez sınıfı):** hangi fonksiyonlar arasında arıyoruz? Burada doğrusal fonksiyonlar.
2. **Kayıp:** bir fonksiyonun ne kadar kötü olduğunu nasıl ölçüyoruz? Burada ortalama kare hata.
3. **Optimizasyon:** en iyi fonksiyonu nasıl buluyoruz? Burada iki yol: kapalı form ve gradyan inişi.

Doğrusal regresyon bu üçlünün en basit, tamamen anlaşılabilir örneğidir. Sinir ağının son katmanı bir doğrusal modeldir; gradyan inişinin bütün davranışları (öğrenme oranı, kararlılık, koşul sayısı, ölçekleme) burada açıkça hesaplanabilir.

### Model {#model}

$$\hat y=w^\top x+b=\sum_{j=1}^dw_jx_j+b.$$
$w\in\mathbb R^d$ **ağırlıklar** (her özniteliğin etkisi), $b$ **sabit terim** (bias, kesişim). Tek öznitelikte bu bir doğrudur; $d=2$'de bir düzlem; genelde bir **hiperdüzlem**.

**Matris biçimi.** $n$ örneği satır olarak dizersek (M0.2: satır = örnek) $X\in\mathbb R^{n\times d}$ ve
$$\hat y=Xw+b\mathbf 1\qquad(n),$$
$b$ her satıra yayınlanır. **Bias hilesi:** $X$'e sonuna 1'lerden bir sütun ekleyip $\tilde X=[X\ \mathbf 1]$, $\theta=(w,b)$ yazınca $\hat y=\tilde X\theta$: bias sıradan bir ağırlık olur. Formüller sadeleşir; kodda genelde ayrı tutulur.

:::yanılgı "Doğrusal regresyon yalnız düz çizgi uydurur"
"Doğrusal", modelin **parametrelere** göre doğrusal olması demektir, girdiye göre değil. $\hat y=w_0+w_1x+w_2x^2+w_3\sin x$ da doğrusal regresyondur: öznitelikleri $\phi(x)=(1,x,x^2,\sin x)$ olarak tanımla, model $w^\top\phi(x)$. Aşağıdaki tüm formüller $X$ yerine $\Phi$ ile aynen geçerlidir. Sinir ağları bu fikrin genellemesidir: $\phi$'yi de öğrenirler.
:::

### Kayıp: ortalama kare hata {#kayip}

$$L(w,b)=\frac1n\sum_{i=1}^n\big(\underbrace{w^\top x_i+b-y_i}_{e_i:\ \text{artık}}\big)^2=\frac1n\|Xw+b\mathbf1-y\|^2.$$

**Neden kare?** Üç gerekçe, en önemlisi sonuncusu:
- Pozitif ve negatif hatalar birbirini götürmez; büyük hatalar orantısız cezalandırılır.
- Türevlenebilir ve dışbükeydir: tek bir minimum, kapalı form çözüm.
- **Olasılıksal:** $y=w^\top x+b+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma^2)$ varsayımı altında MSE'yi küçültmek, en çok olabilirlik kestirimidir (M0.4'teki türetim). Yani MSE seçmek "gürültü Gauss'tur" demektir; gürültü ağır kuyrukluysa (aykırı değerler) bu varsayım bozulur (aşağıda).

**Kayıp yüzeyi.** Tek öznitelikte $L(w,b)$, $(w,b)$ düzleminde bir **çanaktır** (ikinci dereceden bir fonksiyon). Eş-yükselti eğrileri elipstir; elipslerin şekli verinin dağılımına bağlıdır ve gradyan inişinin hızını belirler.

### Gradyan {#gradyan}

:::türetim ∇L = (2/n) Xᵀe
$e=Xw+b\mathbf1-y$ ve $L=\frac1n e^\top e$. Zincir kuralı (M0.3): $\frac{\partial L}{\partial e}=\frac2ne$; $\frac{\partial e}{\partial w}=X$ (Jacobian, $n\times d$). Dolayısıyla
$$\nabla_wL=\Big(\frac{\partial e}{\partial w}\Big)^{\!\top}\frac{\partial L}{\partial e}=\frac2nX^\top e,\qquad \frac{\partial L}{\partial b}=\frac2n\mathbf1^\top e=\frac2n\sum_ie_i.$$
**Shape kontrolü:** $X^\top$ $(d\times n)$, $e$ $(n)$ → $(d)$ = $w$'nin shape'i ✓. **Yorum:** gradyan, her örneğin öznitelik vektörünün kendi hatasıyla ağırlıklandırılmış toplamıdır: $\nabla_wL=\frac2n\sum_ie_ix_i$. Hatası büyük örnekler daha çok çeker; hatası sıfır olanlar hiç çekmez.
:::

### Kapalı form: normal denklemler {#kapali}

Dışbükey ve türevlenebilir bir fonksiyonun minimumunda gradyan sıfırdır. $\tilde X$ ile:
$$\nabla_\theta L=\frac2n\tilde X^\top(\tilde X\theta-y)=0\iff\boxed{\tilde X^\top\tilde X\,\theta=\tilde X^\top y}.$$
Bunlar **normal denklemler**dir; $\tilde X^\top\tilde X$ tersinirse $\hat\theta=(\tilde X^\top\tilde X)^{-1}\tilde X^\top y$.

:::türetim Tek öznitelikte açık formül
$d=1$ için normal denklemleri çözünce: $\hat w=\dfrac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2}=\dfrac{\mathrm{Cov}(x,y)}{\mathrm{Var}(x)}$, $\hat b=\bar y-\hat w\bar x$. Doğru her zaman $(\bar x,\bar y)$ noktasından geçer. $x$'lerin hepsi aynıysa ($\mathrm{Var}(x)=0$) eğim tanımsızdır: veri eğimi belirlemeye yetmez.
:::

**Geometrik anlam: izdüşüm.** $\tilde X\theta$, $\tilde X$'in sütunlarının bir doğrusal birleşimidir (M0.2, sütun görünümü), yani **sütun uzayında** bir vektördür. MSE'yi küçültmek, $y$'ye sütun uzayındaki **en yakın** noktayı bulmaktır. M0.1'de bir vektörün bir doğruya en yakın noktasının **dik izdüşüm** olduğunu ve artığın o doğruya dik olduğunu kanıtlamıştık. Aynısı burada: normal denklemler $\tilde X^\top(y-\tilde X\hat\theta)=0$ demektir, yani **artık vektör her sütuna diktir**.

<figure class="fig"><svg viewBox="0 0 420 250" role="img" aria-label="Sütun uzayı bir düzlem; y düzlemin dışında; ŷ onun düzleme dik izdüşümü; artık düzleme dik"><defs><marker id="mr1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="mr2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cp-f"/></marker><marker id="mr3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><polygon points="30,200 250,230 400,160 180,130" class="soft-a"/><polygon points="30,200 250,230 400,160 180,130" class="ca" fill="none"/><text x="300" y="215" class="sm ta">sütun uzayı: tüm Xθ'lar</text><line x1="120" y1="190" x2="260" y2="60" class="ink w2" marker-end="url(#mr1)"/><text x="266" y="58">y</text><line x1="120" y1="190" x2="260" y2="175" class="cp w2" marker-end="url(#mr2)"/><text x="262" y="195" class="tp">ŷ = Xθ̂</text><line x1="260" y1="175" x2="260" y2="66" class="cb w2 dash" marker-end="url(#mr3)"/><text x="268" y="125" class="tb">e = y − ŷ</text><polyline points="260,163 248,162 248,174" fill="none" class="ink"/><line x1="120" y1="190" x2="190" y2="150" class="cq w2"/><circle cx="190" cy="150" r="3" class="cq-f"/><line x1="190" y1="150" x2="258" y2="64" class="cq dash"/><text x="96" y="140" class="sm tq">Xθ (başka bir seçim)</text><text x="160" y="95" class="sm tq">daha uzun</text></svg><figcaption>Şekil 1 — En küçük kareler, y'nin sütun uzayına dik izdüşümüdür. Artık e her sütuna diktir (Xᵀe = 0); sütun uzayındaki başka herhangi bir nokta y'ye daha uzaktır (Pisagor).</figcaption></figure>

:::kanıt İzdüşümün en yakın nokta olduğu
$\hat y=\tilde X\hat\theta$ normal denklemleri sağlasın, $\tilde X\theta$ sütun uzayında başka bir nokta. $y-\tilde X\theta=(y-\hat y)+(\hat y-\tilde X\theta)$. İkinci terim sütun uzayındadır ($\tilde X(\hat\theta-\theta)$), birinci terim ($e$) sütun uzayına diktir. Pisagor: $\|y-\tilde X\theta\|^2=\|e\|^2+\|\tilde X(\hat\theta-\theta)\|^2\ge\|e\|^2$. Eşitlik ancak $\tilde X\theta=\hat y$ iken. ∎
:::

**Ne zaman tek çözüm vardır?** $\tilde X^\top\tilde X$ ($d+1$ × $d+1$) tersinirdir ⇔ $\tilde X$'in sütunları doğrusal bağımsızdır (tam sütun rank). Bozulduğu durumlar: $n<d+1$ (öznitelikten az örnek), tekrar eden ya da birbirinin doğrusal birleşimi olan öznitelikler (ör. hem santimetre hem inç). O zaman sonsuz çözüm vardır; `lstsq` aralarından **en küçük normlu** olanı döndürür (sözde ters, SVD ile). Düzenlileştirme (M1.3, ridge) $\tilde X^\top\tilde X+\lambda I$ ile bu sorunu ortadan kaldırır.

:::uyarı Tersini açıkça alma
$(\tilde X^\top\tilde X)^{-1}$'i `inv` ile hesaplamak hem yavaş hem sayısal olarak kötüdür: $\tilde X^\top\tilde X$'in koşul sayısı $\tilde X$'inkinin **karesidir** ($\kappa(X^\top X)=\kappa(X)^2$); $\kappa(X)=10^5$ ise float64'te bile ~6 basamak kaybedilir. `torch.linalg.lstsq` / `np.linalg.lstsq` QR ya da SVD ile doğrudan $\tilde X$ üzerinde çalışır. Maliyet $O(nd^2)$ — $d$ binlerceyse ve $n$ milyonlarca ise gradyan yöntemleri tercih edilir.
:::

### Gradyan inişi {#gd}

$$\theta_{t+1}=\theta_t-\eta\,\nabla_\theta L(\theta_t).$$
$\eta$ **öğrenme oranı**dır. Neden işe yarar: doğrusal yaklaşımla (M0.3) $L(\theta-\eta g)\approx L(\theta)-\eta\|g\|^2$; küçük $\eta$ ile kayıp azalır. Ne kadar küçük?

:::türetim Kararlılık koşulu η < 2/λ_max ve yakınsama hızı
MSE ikinci dereceden olduğu için Taylor açılımı **kesindir**: $L(\theta)=L(\theta^*)+\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*)$, $H=\frac2n\tilde X^\top\tilde X$ (sabit Hessian). Gradyan $H(\theta-\theta^*)$. Hata $\delta_t=\theta_t-\theta^*$ için
$$\delta_{t+1}=(I-\eta H)\,\delta_t.$$
$H$'nin özvektörleri yönünde bu, her yönde ayrı bir 1-B problemdir (M1.2'deki $x\leftarrow(1-\eta a)x$): $\lambda_k$ özdeğerli yönde hata her adımda $(1-\eta\lambda_k)$ katına iner.
- **Kararlılık:** her yön için $|1-\eta\lambda_k|<1$ ⇔ $0<\eta<2/\lambda_{\max}$.
- **Hız:** en yavaş yön $\lambda_{\min}$'dir. En iyi sabit $\eta=\frac2{\lambda_{\max}+\lambda_{\min}}$ ile daralma oranı $\frac{\kappa-1}{\kappa+1}$, $\kappa=\lambda_{\max}/\lambda_{\min}$ **koşul sayısı**. $\kappa=100$ ise her adımda hata yalnız %2 azalır: $10^{-6}$'ya inmek ~700 adım.
**Koşul:** $H$ pozitif tanımlı ($\lambda_{\min}>0$). Değilse (rank eksik) GD yine yakınsar ama başlangıca bağlı bir çözüme.
:::

**Öznitelik ölçekleme neden bu kadar önemli?** Tek öznitelikte $H=\frac2n\begin{bmatrix}\sum x_i^2&\sum x_i\\\sum x_i&n\end{bmatrix}$. $x$'ler sıfırdan uzakta toplanmışsa (ör. hep 100 civarı) $w$'yi biraz artırıp $b$'yi biraz azaltmak neredeyse aynı tahmini verir: iki parametre aynı işi yapmaya çalışır, kayıp vadisi çok uzun ve dar olur, $\kappa$ patlar. **Standartlaştırma** $x'=(x-\bar x)/s$ ile $H=2I$, $\kappa=1$: GD tek adımda bile ($\eta=1/2$) optimuma gidebilir. Çok öznitelikte öznitelikler farklı birimlerdeyse (metrekare ve oda sayısı) aynı sorun eksenler arasında olur.

Aşağıda gör: merkezi büyüt, vadinin uzadığını ve GD yolunun zikzak yaptığını izle; sonra standartlaştır.

::lab l1

**Mini-batch ve SGD.** Gradyan $n$ örneğin ortalamasıdır; büyük veride her adımda hepsini kullanmak pahalıdır. **Stokastik gradyan inişi** her adımda rastgele bir mini-batch'in gradyanını kullanır: beklenen değeri tam gradyandır (yansız), gürültüsü $1/\sqrt B$ ile azalır (M0.4). Bu gürültünün davranışını ve momentum, Adam gibi iyileştirmeleri M1.2'de inceleyeceğiz.

### Model nasıl değerlendirilir {#degerlendirme}

- **Artıklar** $e_i=\hat y_i-y_i$: model doğruysa $x$'e karşı çizildiğinde desensiz bir bulut olmalıdır. Eğri bir desen → eksik öznitelik (ör. $x^2$); hunimsi yayılım → gürültü varyansı $x$'e bağlı.
- **$R^2=1-\frac{\sum e_i^2}{\sum(y_i-\bar y)^2}$**: yalnız ortalamayı tahmin eden modele göre açıklanan varyans oranı. Eğitim verisinde öznitelik eklemek $R^2$'yi asla düşürmez; bu yüzden **ayrı bir test kümesinde** değerlendirmek şarttır (M1.3, aşırı uyum).
- **RMSE** $=\sqrt{\text{MSE}}$: $y$ ile aynı birimdedir, yorumlaması kolaydır.

:::ayrıntı Tahminin istatistiksel özellikleri
Model doğruysa ($y=X\theta^*+\varepsilon$, $\varepsilon$ bağımsız, ortalama 0, varyans $\sigma^2$): $\mathbb E[\hat\theta]=\theta^*$ (yansız) ve $\mathrm{Cov}(\hat\theta)=\sigma^2(\tilde X^\top\tilde X)^{-1}$. Öznitelikler birbirine çok benziyorsa (çoklu bağlantı) $(\tilde X^\top\tilde X)^{-1}$ büyük olur: tahminler çok değişken, işaretleri bile güvenilmez — ama tahmin edilen $\hat y$ yine iyi olabilir. **Gauss–Markov teoremi:** bu koşullarda en küçük kareler, tüm doğrusal yansız kestiriciler arasında en küçük varyanslı olandır. Yanlı ama daha düşük varyanslı kestiriciler (ridge) yine de toplam hatada daha iyi olabilir (yan–varyans takası, M2.2).
:::

### Aykırı değerler ve kayıp seçimi {#aykiri}

MSE'nin Gauss varsayımı aykırı değerlerde bozulur: artığı 10 olan tek bir örneğin kayba katkısı 100'dür, gradyana katkısı ise artıkla orantılı ($2e_i$) — sınırsız. Alternatifler:

| kayıp | $\rho(r)$ | gradyan katkısı $\psi(r)$ | varsayılan gürültü | sonuç |
|---|---|---|---|---|
| MSE | $r^2$ | $2r$ | Gauss | koşullu **ortalama** |
| MAE | $\lvert r\rvert$ | $\mathrm{sign}(r)$ | Laplace | koşullu **medyan** |
| Huber | $r^2/2$ ($\lvert r\rvert\le\delta$), $\delta(\lvert r\rvert-\delta/2)$ | $r$ ya da $\pm\delta$ | merkez Gauss, kuyruk Laplace | ikisinin arası |

`nn.HuberLoss` ve `nn.SmoothL1Loss` (nesne tespitinde kutu regresyonu, M4.9) bu ailedendir. Aşağıda aykırı oranını artır ve üç kaybın doğruyu nasıl farklı çektiğini gör.

::lab l2

### PyTorch'ta doğrusal regresyon {#pytorch}

```python
import torch, torch.nn as nn
model = nn.Linear(d, 1)                 # weight: (1, d), bias: (1,)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(500):
    pred = model(X).squeeze(-1)          # (n, 1) → (n,)   ← shape'e dikkat (M0.2)
    loss = loss_fn(pred, y)
    opt.zero_grad()                      # .grad birikir, sıfırla (M0.3)
    loss.backward()                      # ∇ = (2/n) Xᵀe, autograd hesaplar
    opt.step()                           # θ ← θ − η ∇
```

Bu beş satır, bu kursun geri kalanındaki **her** eğitim döngüsünün iskeletidir; değişen yalnız model, kayıp ve optimizer'dır. Kapalı form için `torch.linalg.lstsq(Xt, y).solution`. `squeeze` unutulursa `pred` (n, 1), `y` (n,) olur ve `(pred - y)` yayınla (n, n) bir matrise döner — `MSELoss` bu durumda uyarı verir; elle yazılmış kayıpta hata sessizdir.

### Özet {#ozet}

| | kapalı form | gradyan inişi |
|---|---|---|
| hesap | $\tilde X^\top\tilde X\theta=\tilde X^\top y$, $O(nd^2+d^3)$ | adım başına $O(nd)$, adım sayısı ~$\kappa$ |
| kesinlik | tek seferde kesin (yuvarlama dışında) | yaklaşık, $\eta$ ve adım sayısına bağlı |
| genelleşir mi | yalnız doğrusal + kare kayıp | her türevlenebilir model ve kayıp |
| zayıf nokta | rank eksikliği, $\kappa^2$ | koşul sayısı, $\eta$ seçimi |

**Akılda kalsın:** MSE = Gauss gürültüsü varsayımı. En küçük kareler çözümü $y$'nin sütun uzayına dik izdüşümüdür; artık her özniteliğe diktir. GD'nin kararlılık sınırı $2/\lambda_{\max}$, hızı koşul sayısına bağlıdır; öznitelikleri standartlaştırmak koşul sayısını düşürür. Aykırı değerler varsa kayıp seçimini yeniden düşün.

**Kaynaklar:** Stanford CS229 ders notları (en küçük kareler, normal denklemler, olasılıksal yorum); James, Witten, Hastie, Tibshirani, *An Introduction to Statistical Learning (Python)*, Bölüm 3; Strang, *Introduction to Linear Algebra* Bölüm 4.2–4.3 (izdüşüm ve en küçük kareler); Boyd ve Vandenberghe, *Convex Optimization* §9.3 (gradyan inişinin yakınsama analizi); Huber (1964), "Robust estimation of a location parameter"; d2l.ai §3.1–3.5; PyTorch `nn.Linear`, `torch.linalg.lstsq` belgeleri.
