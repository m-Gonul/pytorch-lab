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
- **Olasılıksal:** $y=w^\top x+b+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma^2)$ varsayımı altında MSE'yi küçültmek, en çok olabilirlik kestirimidir (aşağıdaki türetim). Yani MSE seçmek "gürültü Gauss'tur" demektir; gürültü ağır kuyrukluysa (aykırı değerler) bu varsayım bozulur (aşağıda).

:::türetim MSE = Gauss gürültüsü altında en çok olabilirlik
Model $y_i=w^\top x_i+b+\varepsilon_i$, $\varepsilon_i\sim\mathcal N(0,\sigma^2)$ bağımsız olsun. Verinin olabilirliği (M0.4: bağımsızlıkla çarpım):
$$p(y\mid X;\theta)=\prod_i\frac{1}{\sqrt{2\pi}\,\sigma}\exp\Big(-\frac{(w^\top x_i+b-y_i)^2}{2\sigma^2}\Big).$$
Log'unu al:
$$\log p(y\mid X;\theta)=-\frac{1}{2\sigma^2}\sum_i e_i^2-n\log(\sqrt{2\pi}\,\sigma).$$
İkinci terim $\theta$'ya bağlı değildir; birinciyi en büyüklemek, $\sum_i e_i^2$'yi — yani MSE'yi — en küçüklemektir. $\sigma^2$'nin $\theta$'ya bağlı olmaması bu sonucu değiştirmez. Yorum: **MSE'nin gerekçesi bir optimizasyon keyfi değil, bir istatistiksel varsayımdır.** Gürültü dağılımını değiştirirsen kayıp değişir: Laplace gürültüsü $\log p=-\frac1b\sum_i\lvert e_i\rvert+\cdots$ verir, yani MAE (aşağıda); merkezi Gauss + ağır kuyruk karışımı Huber'i gerekçelendirir. Kayıp tablosunun en sağdaki sütunu işte bu okumadır.
:::

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

<figure class="fig"><svg viewBox="0 0 420 250" role="img" aria-label="Sütun uzayı bir düzlem; y düzlemin dışında; ŷ onun düzleme dik izdüşümü; artık düzleme dik"><defs><marker id="mr1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="mr2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cp-f"/></marker><marker id="mr3" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><polygon points="30,200 250,230 400,160 180,130" class="soft-a"/><polygon points="30,200 250,230 400,160 180,130" class="ca" fill="none"/><text x="300" y="215" class="sm ta">sütun uzayı: tüm Xθ'lar</text><line x1="120" y1="190" x2="260" y2="60" class="ink w2" marker-end="url(#mr1)"/><text x="266" y="58">y</text><line x1="120" y1="190" x2="260" y2="175" class="cp w2" marker-end="url(#mr2)"/><text x="262" y="195" class="tp">ŷ = Xθ<tspan dx="-0.44em" dy="-0.2em">ˆ</tspan><tspan dx="0.09em" dy="0.2em"> </tspan></text><line x1="260" y1="175" x2="260" y2="66" class="cb w2 dash" marker-end="url(#mr3)"/><text x="268" y="125" class="tb">e = y − ŷ</text><polyline points="260,163 248,162 248,174" fill="none" class="ink"/><line x1="120" y1="190" x2="190" y2="150" class="cq w2"/><circle cx="190" cy="150" r="3" class="cq-f"/><line x1="190" y1="150" x2="258" y2="64" class="cq dash"/><text x="96" y="140" class="sm tq">Xθ (başka bir seçim)</text><text x="160" y="95" class="sm tq">daha uzun</text></svg><figcaption>Şekil 1 — En küçük kareler, y'nin sütun uzayına dik izdüşümüdür. Artık e her sütuna diktir (Xᵀe = 0); sütun uzayındaki başka herhangi bir nokta y'ye daha uzaktır (Pisagor).</figcaption></figure>

:::kanıt İzdüşümün en yakın nokta olduğu
$\hat y=\tilde X\hat\theta$ normal denklemleri sağlasın, $\tilde X\theta$ sütun uzayında başka bir nokta. $y-\tilde X\theta=(y-\hat y)+(\hat y-\tilde X\theta)$. İkinci terim sütun uzayındadır ($\tilde X(\hat\theta-\theta)$), birinci terim ($e$) sütun uzayına diktir. Pisagor: $\|y-\tilde X\theta\|^2=\|e\|^2+\|\tilde X(\hat\theta-\theta)\|^2\ge\|e\|^2$. Eşitlik ancak $\tilde X\theta=\hat y$ iken. ∎
:::

Kapalı formun bu davranışını canlı görmek için aşağıdaki deneyde noktaları sürükle: doğrunun her zaman ağırlık merkezi $(\bar x,\bar y)$'den geçtiğini ve bir aykırı değerin doğruyu ne kadar çekebildiğini sen de üretebilirsin.

::lab l3

**Ne zaman tek çözüm vardır?** $\tilde X^\top\tilde X$ ($d+1$ × $d+1$) tersinirdir ⇔ $\tilde X$'in sütunları doğrusal bağımsızdır (tam sütun rank). Bozulduğu durumlar: $n<d+1$ (öznitelikten az örnek), tekrar eden ya da birbirinin doğrusal birleşimi olan öznitelikler (ör. hem santimetre hem inç). O zaman sonsuz çözüm vardır; `lstsq` aralarından **en küçük normlu** olanı döndürür (sözde ters, SVD ile). Düzenlileştirme (M1.3, ridge) $\tilde X^\top\tilde X+\lambda I$ ile bu sorunu ortadan kaldırır.

:::uyarı Tersini açıkça alma
$(\tilde X^\top\tilde X)^{-1}$'i `inv` ile hesaplamak hem yavaş hem sayısal olarak kötüdür: $\tilde X^\top\tilde X$'in koşul sayısı $\tilde X$'inkinin **karesidir** ($\kappa(X^\top X)=\kappa(X)^2$); $\kappa(X)=10^5$ ise float64'te bile ~6 basamak kaybedilir. `torch.linalg.lstsq` / `np.linalg.lstsq` QR ya da SVD ile doğrudan $\tilde X$ üzerinde çalışır. Maliyet $O(nd^2)$ — $d$ binlerceyse ve $n$ milyonlarca ise gradyan yöntemleri tercih edilir.
:::

### Öznitelik genişletme: formüller hiç değişmez {#oznitelik}

Yukarıdaki yanılgı kutusunun kod karşılığı: $\hat y=w_0+w_1x+w_2x^2$ uydurmak istiyorsun ama her formül tek doğrunun. Çözüm, özniteliği vektöre çevirmek: $\phi(x)=[1,\,x,\,x^2]$, $\Phi\in\mathbb R^{n\times3}$'ın satırı $\phi(x_i)$. Model $\hat y=w^\top\phi(x)$ — **parametrelerde hâlâ doğrusal**; bu bölümdeki her şey aynen geçerli, yalnız $X$ yerine $\Phi$ yaz:
$$\nabla_wL=\frac2n\Phi^\top e,\qquad \Phi^\top\Phi\,w=\Phi^\top y,\qquad \hat y=\Phi w.$$
İzdüşüm yorumu da aynıdır: en iyi $y$ tahmini, $\Phi$'nın sütun uzayındaki (tüm $(1,x,x^2)$ kombinasyonlarının, yani tüm parabollerin) düzlemine dik izdüşümdür. Tek fark pratik: $1$, $x$, $x^2$ sütunları birbirine benzer (hepsi $x$'in aynı bölgede büyür), $\kappa(\Phi)$ büyür; bu yüzden ölçekleme burada daha da önemli. Aşağıdaki E1 alıştırmasında bunu NumPy ile kodlayacak, P5'teki U şekilli artıkların kaybolmasını ölçeceksin.

### Gradyan inişi {#gd}

$$\theta_{t+1}=\theta_t-\eta\,\nabla_\theta L(\theta_t).$$
$\eta$ **öğrenme oranı**dır. Neden işe yarar: doğrusal yaklaşımla (M0.3) $L(\theta-\eta g)\approx L(\theta)-\eta\|g\|^2$; küçük $\eta$ ile kayıp azalır. Ne kadar küçük?

:::türetim Kararlılık koşulu η < 2/λ_max ve yakınsama hızı
MSE ikinci dereceden olduğu için Taylor açılımı **kesindir**: $L(\theta)=L(\theta^*)+\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*)$, $H=\frac2n\tilde X^\top\tilde X$ (sabit Hessian). Gradyan $H(\theta-\theta^*)$. Hata $\delta_t=\theta_t-\theta^*$ için
$$\delta_{t+1}=(I-\eta H)\,\delta_t.$$
$H$'nin özvektörleri yönünde bu, her yönde ayrı bir 1-B problemdir (M1.2'deki $x\leftarrow(1-\eta a)x$): $\lambda_k$ özdeğerli yönde hata her adımda $(1-\eta\lambda_k)$ katına iner.
- **Kararlılık:** her yön için $|1-\eta\lambda_k|<1$ ⇔ $0<\eta<2/\lambda_{\max}$.
- **Hız:** en yavaş yön $\lambda_{\min}$'dir. En iyi sabit $\eta=\frac2{\lambda_{\max}+\lambda_{\min}}$ ile daralma oranı $\frac{\kappa-1}{\kappa+1}$, $\kappa=\lambda_{\max}/\lambda_{\min}$ **koşul sayısı**. $\kappa=100$ ise her adımda hata yalnız %2 azalır: $10^{-6}$'ya inmek ~700 adım.

Adım sayısını koşul sayısının fonksiyonu olarak betikle hesaplayıp tabloya dökmek, büyümeyi ellinden izler:

| koşul sayısı $\kappa$ | adım başına daralma $\frac{\kappa-1}{\kappa+1}$ | hatayı $10^{-6}$'ya indirmek ($\eta$ en iyi) |
|---|---|---|
| 1 (standartlaştırılmış) | 0 | 1 adım ($\eta=\tfrac12$, $H=2I$) |
| 10 | 0.818 | ≈ 69 adım |
| 100 | 0.980 | ≈ 691 adım |
| 10 000 | 0.9998 | ≈ 69 000 adım |

Adım sayısı $\kappa\ln 10$ ile lineer büyür: koşul sayısında her bir ondalık, ~10 kat fazla GD adımı demektir. Öğrenme oranını sabit tutmak yerine koşul sayısını küçültmek (ölçekleme, M1.2'de momentum) geçerli çözümdür.
**Koşul:** $H$ pozitif tanımlı ($\lambda_{\min}>0$). Değilse (rank eksik) GD yine yakınsar ama başlangıca bağlı bir çözüme.
:::

<figure class="fig"><svg viewBox="0 0 420 250" role="img" aria-label="Kayıp yüzeyinin eş-yükselti elipsleri ve gradyan inişi yolu: vadi uzundur, yol elipslere dik zikzak yapar"><defs><marker id="f3a" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><line x1="40" y1="79.28571428571429" x2="400" y2="79.28571428571429" class="mute"/><line x1="66.66666666666666" y1="215" x2="66.66666666666666" y2="15" class="mute"/><text x="400" y="93.28571428571429" class="sm">w</text><text x="58.66666666666666" y="23" class="sm">b</text><path d="M355.0,112.7 L367.2,124.6 L377.6,136.4 L386.2,147.7 L392.7,158.6 L397.1,168.6 L399.2,177.8 L399.1,185.9 L396.7,192.7 L392.0,198.3 L385.3,202.4 L376.5,205.0 L365.8,206.1 L353.4,205.6 L339.6,203.6 L324.5,200.0 L308.4,195.0 L291.5,188.7 L274.3,181.0 L257.0,172.3 L239.8,162.5 L223.0,152.0 L207.0,140.8 L192.1,129.2 L178.4,117.3 L166.2,105.4 L155.7,93.6 L147.1,82.3 L140.6,71.4 L136.3,61.4 L134.1,52.2 L134.3,44.1 L136.7,37.3 L141.3,31.7 L148.1,27.6 L156.9,25.0 L167.5,23.9 L179.9,24.4 L193.8,26.4 L208.9,30.0 L225.0,35.0 L241.8,41.3 L259.0,49.0 L276.4,57.7 L293.6,67.5 L310.3,78.0 L326.3,89.2 L341.3,100.8 L355.0,112.7" class="cq" fill="none"/><path d="M322.5,113.5 L330.2,121.1 L336.8,128.5 L342.3,135.7 L346.4,142.5 L349.1,148.9 L350.5,154.7 L350.4,159.8 L348.9,164.2 L346.0,167.7 L341.7,170.3 L336.1,171.9 L329.4,172.6 L321.5,172.3 L312.8,171.0 L303.2,168.8 L293.0,165.6 L282.4,161.6 L271.5,156.8 L260.5,151.2 L249.6,145.1 L239.1,138.4 L228.9,131.3 L219.5,124.0 L210.8,116.5 L203.1,108.9 L196.5,101.5 L191.1,94.3 L187.0,87.5 L184.2,81.1 L182.8,75.3 L182.9,70.2 L184.4,65.8 L187.4,62.3 L191.7,59.7 L197.2,58.1 L204.0,57.4 L211.8,57.7 L220.6,59.0 L230.1,61.2 L240.3,64.4 L250.9,68.4 L261.8,73.2 L272.8,78.8 L283.7,84.9 L294.3,91.6 L304.4,98.7 L313.9,106.0 L322.5,113.5" class="cq" fill="none"/><path d="M294.6,114.3 L298.4,118.0 L301.8,121.8 L304.5,125.4 L306.5,128.8 L307.9,132.0 L308.6,134.9 L308.5,137.4 L307.8,139.6 L306.3,141.3 L304.2,142.6 L301.4,143.5 L298.0,143.8 L294.1,143.7 L289.7,143.0 L284.9,141.9 L279.9,140.3 L274.5,138.3 L269.1,135.9 L263.6,133.1 L258.2,130.0 L252.9,126.7 L247.8,123.2 L243.1,119.5 L238.7,115.7 L234.9,112.0 L231.6,108.2 L228.9,104.6 L226.8,101.2 L225.4,98.0 L224.8,95.1 L224.8,92.6 L225.6,90.4 L227.0,88.7 L229.2,87.4 L231.9,86.5 L235.3,86.2 L239.2,86.3 L243.6,87.0 L248.4,88.1 L253.5,89.7 L258.8,91.7 L264.2,94.1 L269.7,96.9 L275.2,100.0 L280.5,103.3 L285.5,106.8 L290.3,110.5 L294.6,114.3" class="cq" fill="none"/><text x="360" y="195.35714285714286" class="sm tq">L = sabit (elipsler)</text><line x1="147.26192883478382" y1="5.280877456118805" x2="386.0714044985495" y2="224.7191225438812" class="ca dash"/><line x1="210.92917214918802" y1="116.45068087524606" x2="322.4041611841452" y2="113.54931912475394" class="ca dash"/><text x="352" y="166.7857142857143" class="sm ta">λ_min yönü: uzun vadi</text><text x="273.33333333333337" y="126.42857142857143" class="sm ta">λ_max yönü: dik duvar</text><path d="M66.7,79.3 L318.7,61.8 L155.1,75.1 L264.3,68.1 L194.4,74.4 L242.0,72.0 L212.5,75.2 L233.6,74.7 L221.4,76.6" class="cb w2" marker-end="url(#f3a)"/><circle cx="221.36694191327913" cy="76.62312142729509" r="2.5" class="cb-f"/><circle cx="66.66666666666666" cy="79.28571428571429" r="3" class="ink-f"/><text x="73.66666666666666" y="72.28571428571429" class="sm">başlangıç (0,0)</text><circle cx="266.66666666666663" cy="115" r="3" class="cp-f"/><text x="273.66666666666663" y="129" class="sm tp">optimum (w*, b*)</text></svg><figcaption>Şekil 2 — Tek öznitelikte MSE'nin Hessian'ı sabittir: <span class="hand-font">H = (2/n)·X̃ᵀX̃ = [[22, 6], [6, 2]]</span>, özdeğerleri λ_max = 22.58 ve λ_min = 1.42 (κ ≈ 16; veri x = [1…5], y = 1.5x − 1). Eş-yükselti eğrileri bu eksenlere hizalı elipslerdir; η = 0.07 &lt; 2/λ_max = 0.089 ile (0,0)'dan başlayan GD her adımda elipse dik (λ_max yönü) sert adım atıp vadi içinde zikzak yapar. Sayılar betikle hesaplandı; etkileşimli sürüm aşağıdaki deneyde.</figcaption></figure>

**Öznitelik ölçekleme neden bu kadar önemli?** Tek öznitelikte $H=\frac2n\begin{bmatrix}\sum x_i^2&\sum x_i\\\sum x_i&n\end{bmatrix}$. $x$'ler sıfırdan uzakta toplanmışsa (ör. hep 100 civarı) $w$'yi biraz artırıp $b$'yi biraz azaltmak neredeyse aynı tahmini verir: iki parametre aynı işi yapmaya çalışır, kayıp vadisi çok uzun ve dar olur, $\kappa$ patlar. **Standartlaştırma** $x'=(x-\bar x)/s$ ile $H=2I$, $\kappa=1$: GD tek adımda bile ($\eta=1/2$) optimuma gidebilir. Çok öznitelikte öznitelikler farklı birimlerdeyse (metrekare ve oda sayısı) aynı sorun eksenler arasında olur.

Aşağıda gör: merkezi büyüt, vadinin uzadığını ve GD yolunun zikzak yaptığını izle; sonra standartlaştır.

::lab l1

**Mini-batch ve SGD.** Gradyan $n$ örneğin ortalamasıdır; büyük veride her adımda hepsini kullanmak pahalıdır. **Stokastik gradyan inişi** her adımda rastgele bir mini-batch'in gradyanını kullanır: beklenen değeri tam gradyandır (yansız), gürültüsü batch küçüldükçe büyür — $B$ örneklik bir batch için
$$\mathbb E[\hat g_B]=g,\qquad \mathrm{Var}(\hat g_B)=\frac{\mathrm{Var}(g_i)}{B}$$
(M0.4: bağımsız örneklerin ortalamasının varyansı). Yani gürültüyü 2 kat azaltmak batch'i 4 kat büyütmektir; aynı hesap bütçesinde 4 kat daha az adım atmak zorunda kalmakla takas edilir. Bu gürültünün davranışı ve momentum, Adam gibi iyileştirmeler M1.2'de.

### Model nasıl değerlendirilir {#degerlendirme}

- **Artıklar** $e_i=\hat y_i-y_i$: model doğruysa $x$'e karşı çizildiğinde desensiz bir bulut olmalıdır. Eğri bir desen → eksik öznitelik (ör. $x^2$); hunimsi yayılım → gürültü varyansı $x$'e bağlı.
- **$R^2=1-\frac{\sum e_i^2}{\sum(y_i-\bar y)^2}$**: yalnız ortalamayı tahmin eden modele göre açıklanan varyans oranı. Eğitim verisinde öznitelik eklemek $R^2$'yi asla düşürmez; bu yüzden **ayrı bir test kümesinde** değerlendirmek şarttır (M1.3, aşırı uyum).
- **RMSE** $=\sqrt{\text{MSE}}$: $y$ ile aynı birimdedir, yorumlaması kolaydır.

<figure class="fig wide"><svg viewBox="0 0 420 280" role="img" aria-label="Anscombe dörlüsü: dört farklı veri kümesi, aynı özet istatistikler, bambaşka desenler"><rect x="30" y="30" width="136" height="86" class="mute" fill="none"/><line x1="30" y1="108.7" x2="166" y2="108.7" class="ca dash"/><text x="8" y="24" class="sm tb">I: desensiz bulut</text><circle cx="90.9" cy="72.1" r="2.6" class="cp-f"/><circle cx="76.7" cy="80.0" r="2.6" class="cp-f"/><circle cx="112.2" cy="75.4" r="2.6" class="cp-f"/><circle cx="83.8" cy="66.5" r="2.6" class="cp-f"/><circle cx="98.0" cy="70.0" r="2.6" class="cp-f"/><circle cx="119.3" cy="58.1" r="2.6" class="cp-f"/><circle cx="62.4" cy="77.9" r="2.6" class="cp-f"/><circle cx="48.2" cy="99.6" r="2.6" class="cp-f"/><circle cx="105.1" cy="51.7" r="2.6" class="cp-f"/><circle cx="69.6" cy="95.5" r="2.6" class="cp-f"/><circle cx="55.3" cy="89.2" r="2.6" class="cp-f"/><text x="166" y="131" class="sm" text-anchor="end">x</text><text x="22" y="38" class="sm">y</text><rect x="235" y="30" width="136" height="86" class="mute" fill="none"/><line x1="235" y1="108.7" x2="371" y2="108.7" class="ca dash"/><text x="213" y="24" class="sm tb">II: parabol</text><circle cx="295.9" cy="64.1" r="2.6" class="cp-f"/><circle cx="281.7" cy="71.3" r="2.6" class="cp-f"/><circle cx="317.2" cy="67.0" r="2.6" class="cp-f"/><circle cx="288.8" cy="66.8" r="2.6" class="cp-f"/><circle cx="303.0" cy="63.2" r="2.6" class="cp-f"/><circle cx="324.3" cy="71.6" r="2.6" class="cp-f"/><circle cx="267.4" cy="86.0" r="2.6" class="cp-f"/><circle cx="253.2" cy="108.0" r="2.6" class="cp-f"/><circle cx="310.1" cy="64.1" r="2.6" class="cp-f"/><circle cx="274.6" cy="77.7" r="2.6" class="cp-f"/><circle cx="260.3" cy="96.1" r="2.6" class="cp-f"/><text x="371" y="131" class="sm" text-anchor="end">x</text><text x="227" y="38" class="sm">y</text><rect x="30" y="170" width="136" height="86" class="mute" fill="none"/><line x1="30" y1="248.7" x2="166" y2="248.7" class="ca dash"/><text x="8" y="164" class="sm tb">III: aykırı değer</text><circle cx="90.9" cy="216.3" r="2.6" class="cp-f"/><circle cx="76.7" cy="221.3" r="2.6" class="cp-f"/><circle cx="112.2" cy="177.9" r="2.6" class="cp-f"/><circle cx="83.8" cy="218.8" r="2.6" class="cp-f"/><circle cx="98.0" cy="213.7" r="2.6" class="cp-f"/><circle cx="119.3" cy="206.3" r="2.6" class="cp-f"/><circle cx="62.4" cy="226.3" r="2.6" class="cp-f"/><circle cx="48.2" cy="231.3" r="2.6" class="cp-f"/><circle cx="105.1" cy="211.3" r="2.6" class="cp-f"/><circle cx="69.6" cy="223.9" r="2.6" class="cp-f"/><circle cx="55.3" cy="228.9" r="2.6" class="cp-f"/><text x="166" y="271" class="sm" text-anchor="end">x</text><text x="22" y="178" class="sm">y</text><rect x="235" y="170" width="136" height="86" class="mute" fill="none"/><line x1="235" y1="248.7" x2="371" y2="248.7" class="ca dash"/><text x="213" y="164" class="sm tb">IV: tekil x</text><circle cx="281.7" cy="222.7" r="2.6" class="cp-f"/><circle cx="281.7" cy="228.7" r="2.6" class="cp-f"/><circle cx="281.7" cy="214.5" r="2.6" class="cp-f"/><circle cx="281.7" cy="206.3" r="2.6" class="cp-f"/><circle cx="281.7" cy="208.9" r="2.6" class="cp-f"/><circle cx="281.7" cy="219.3" r="2.6" class="cp-f"/><circle cx="281.7" cy="232.4" r="2.6" class="cp-f"/><circle cx="359.9" cy="230.1" r="2.6" class="cp-f"/><circle cx="281.7" cy="213.0" r="2.6" class="cp-f"/><circle cx="281.7" cy="220.4" r="2.6" class="cp-f"/><circle cx="281.7" cy="179.6" r="2.6" class="cp-f"/><text x="371" y="271" class="sm" text-anchor="end">x</text><text x="227" y="178" class="sm">y</text></svg><figcaption>Şekil 3 — Anscombe'un dörlüsü (1973). Dört veri kümesinde de x̄ = 9, ȳ = 7.5, en küçük kareler doğrusu ŷ = 0.5x + 3 ve R² = 0.667: özet sayılar özdeş, desenler bambaşka. R² ve katsayılar yalnız başına yetmez; artıkların grafiğine bakmak şart — değerlendirme bölümü bu yüzden artıklarla başlar.</figcaption></figure>

:::ayrıntı Hunimsi yayılım: ağırlıklı en küçük kareler
Artıkların yayılımı $x$'e bağlıysa (küçük $x$'te dar, büyükte geniş — huni), her örnek kayba eşit katkı sunmamalıdır: gürültüsü küçük örneklerin söz hakkı büyük olmalı. Örnek başına gürültü varyansı $v_i$ olsun; $L_w=\frac1n\sum_i v_i^{-1}e_i^2$'yi en küçükleyen çözüm **ağırlıklı normal denklemler**dir:
$$\hat\theta=(\Phi^\top W\Phi)^{-1}\Phi^\top Wy,\qquad W=\mathrm{diag}(1/v_1,\dots,1/v_n).$$
Pratikte $v_i$ bilinmez: önce ham en küçük karelerle uydur, artıkların karesinden $v_i$ kestir, bir kez yeniden ağırlıkla (bir adımlık IRLS — aşağıdaki robustfit deneyinin aynısı). Gauss–Markov çerçevesinde bu, heteroskedastik veride en küçük varyanslı yansız kestiriciyi geri getirir. Sinir ağlarındaki karşılığı örnek ağırlıklamadır (`loss` çağrısında `weight`).
:::

:::ayrıntı Tahminin istatistiksel özellikleri
Model doğruysa ($y=X\theta^*+\varepsilon$, $\varepsilon$ bağımsız, ortalama 0, varyans $\sigma^2$): $\mathbb E[\hat\theta]=\theta^*$ (yansız) ve $\mathrm{Cov}(\hat\theta)=\sigma^2(\tilde X^\top\tilde X)^{-1}$. Öznitelikler birbirine çok benziyorsa (çoklu bağlantı) $(\tilde X^\top\tilde X)^{-1}$ büyük olur: tahminler çok değişken, işaretleri bile güvenilmez — ama tahmin edilen $\hat y$ yine iyi olabilir. **Gauss–Markov teoremi:** bu koşullarda en küçük kareler, tüm doğrusal yansız kestiriciler arasında en küçük varyanslı olandır. Yanlı ama daha düşük varyanslı kestiriciler (ridge) yine de toplam hatada daha iyi olabilir (yan–varyans takası, M2.2).
:::

Aşağıdaki deney, bu bölümün tamamını tek ekranda toplar: üç farklı gerçek modelde (doğrusal, karesel, hunimsi gürültü) doğrusal model uydurur ve artıkları üç biçimde gösterir. Şekil 3'te gördüğün dört deseni burada kendin üret ve artıklardan oku.

::lab l4

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

**Kaynaklar:** Stanford CS229 ders notları (en küçük kareler, normal denklemler, olasılıksal yorum); James, Witten, Hastie, Tibshirani, *An Introduction to Statistical Learning (Python)*, Bölüm 3; Strang, *Introduction to Linear Algebra* Bölüm 4.2–4.3 (izdüşüm ve en küçük kareler); Boyd ve Vandenberghe, *Convex Optimization* §9.3 (gradyan inişinin yakınsama analizi); Huber (1964), "Robust estimation of a location parameter"; Anscombe (1973), "Graphs in Statistical Analysis" (Şekil 3'ün verileri); d2l.ai §3.1–3.5; PyTorch `nn.Linear`, `torch.linalg.lstsq` belgeleri.
