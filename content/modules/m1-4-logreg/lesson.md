### Problem: sınıflandırma {#problem}

Hedef artık sürekli bir sayı değil, bir **sınıf**: e-posta spam mı (1) değil mi (0), tümör iyi huylu mu kötü huylu mu. Doğrusal regresyonu doğrudan uygulamak ($\hat y=w^\top x+b$, eşik 0.5) üç sebeple kötüdür:

1. Çıktı $[0,1]$ dışına taşar; olasılık olarak yorumlanamaz.
2. MSE, doğru tarafta ama sınırdan çok uzak örnekleri de cezalandırır: çok "emin ve doğru" bir örnek sınırı kendine doğru çeker.
3. Gürültü varsayımı (Gauss) 0/1 hedefler için anlamsızdır.

Lojistik regresyon, M0.4'teki tarifi uygular: **önce olasılıksal bir model kur, sonra negatif log-olabilirliği küçült**. Etiket bir yazı-turadır, $y\mid x\sim\text{Bernoulli}(p(x))$; işimiz $p(x)$'i doğrusal bir modelle ifade etmek.

### Logit ve sigmoid {#sigmoid}

Doğrusal bir ifade $z=w^\top x+b\in(-\infty,\infty)$ üretir; olasılık $(0,1)$'dedir. Köprü **log-odds**'tur: $p$ olasılıklı bir olayın **odds**'u (bahis oranı) $\frac p{1-p}\in(0,\infty)$, logaritması $\log\frac p{1-p}\in(-\infty,\infty)$. Lojistik regresyonun varsayımı: **log-odds girdinin doğrusal fonksiyonudur.**
$$\log\frac p{1-p}=z=w^\top x+b\quad\Longleftrightarrow\quad p=\sigma(z)=\frac1{1+e^{-z}}.$$
$z$'ye **logit** denir. Sigmoid'in özellikleri: $\sigma(0)=\frac12$; $\sigma(-z)=1-\sigma(z)$; $\sigma'(z)=\sigma(z)(1-\sigma(z))$ (M0.3'te türettik).

**Katsayıların yorumu:** $x_j$'yi 1 birim artırmak log-odds'u $w_j$ kadar, yani odds'u $e^{w_j}$ **katına** çıkarır. Olasılık üzerindeki etkisi ise $p$'nin nerede olduğuna bağlıdır (sigmoid'in dik orta kısmında büyük, uçlarda küçük).

<figure class="fig"><svg viewBox="0 0 420 220" role="img" aria-label="Sigmoid eğrisi: logit z'yi olasılık p'ye çevirir"><line x1="20" y1="190" x2="400" y2="190" class="mute"/><line x1="210" y1="20" x2="210" y2="200" class="mute"/><line x1="20" y1="30" x2="400" y2="30" class="mute dash"/><line x1="20" y1="110" x2="400" y2="110" class="mute dash"/><path d="M20,189.4 C120,185 160,170 185,140 C200,122 205,115 210,110 C215,105 220,98 235,80 C260,50 300,35 400,30.6" class="ca w3" fill="none"/><line x1="160" y1="130" x2="260" y2="90" class="cb dash"/><text x="214" y="200" class="sm">0</text><text x="396" y="205" class="sm" text-anchor="end">z (logit)</text><text x="24" y="26" class="sm">p = 1</text><text x="24" y="106" class="sm">p = ½</text><text x="268" y="88" class="sm tb">eğim σ′(0) = ¼</text><text x="300" y="60" class="sm ta">σ(z) = 1/(1 + e<tspan baseline-shift="4" font-size="75%">−z</tspan>)</text><text x="40" y="175" class="sm">z ≪ 0: p ≈ e^z</text><text x="310" y="45" class="sm">z ≫ 0: p ≈ 1</text></svg><figcaption>Şekil 1 — Sigmoid, logiti olasılığa çevirir: z = 0'da ½, en dik yeri orası (eğim ¼), uçlarda düzleşir. Düz uçlar, yanlış bir kayıpla (MSE) eğitimin neden durduğunun sebebidir.</figcaption></figure>

### Kayıp: ikili çapraz entropi {#bce}

:::türetim NLL = ikili çapraz entropi
Bernoulli olasılığı: $P(y\mid x)=p^y(1-p)^{1-y}$ ($y=1$'de $p$, $y=0$'da $1-p$). Negatif log:
$$\ell=-\big[y\log p+(1-y)\log(1-p)\big].$$
Tüm veri: $L=\frac1n\sum_i\ell_i$. Bu, $y$'nin "gerçek dağılımı" (bir noktaya yığılmış) ile modelin dağılımı arasındaki **çapraz entropidir** (M0.4).
:::

**Marj biçimi.** $y\in\{0,1\}$ yerine $s=2y-1\in\{-1,+1\}$ yazınca kayıp tek bir ifadeye iner: $\ell=\log(1+e^{-s z})$. $m=sz$ **marjdır**: pozitifse örnek doğru tarafta, büyüklüğü ne kadar emin olduğumuzu ölçer. Lojistik kayıp $m\to\infty$'da 0'a, $m\to-\infty$'da $-m$ gibi (doğrusal) büyür.

:::türetim Gradyan: ∂ℓ/∂z = p − y
$\ell=-y\log\sigma(z)-(1-y)\log(1-\sigma(z))$. $\frac{d}{dz}\log\sigma(z)=1-\sigma(z)$ ve $\frac d{dz}\log(1-\sigma(z))=-\sigma(z)$ (sigmoid türevinden). Böylece
$$\frac{\partial\ell}{\partial z}=-y(1-p)+(1-y)p=p-y.$$
$z=w^\top x+b$ olduğundan $\nabla_w\ell=(p-y)x$, $\partial\ell/\partial b=p-y$; tüm veri için
$$\nabla_wL=\frac1nX^\top(p-y),\qquad\frac{\partial L}{\partial b}=\frac1n\sum_i(p_i-y_i).$$
**Yorum:** doğrusal regresyondaki $\frac2nX^\top(\hat y-y)$ ile aynı biçim: "tahmin eksi hedef, girdiyle ağırlıklandırılmış". Sigmoid'in $\sigma'$ çarpanı log ile **sadeleşti**. Bu tesadüf değildir: kanonik bağlantılı her genelleştirilmiş doğrusal modelde (Gauss–özdeşlik, Bernoulli–logit, kategorik–softmax, Poisson–log) gradyan "tahmin − hedef" çıkar.
:::

**Neden MSE değil?** $\sigma(z)$ üzerinde MSE kullansaydık $\frac{\partial}{\partial z}(p-y)^2=2(p-y)\,p(1-p)$ olurdu. Model **emin ve yanlışsa** ($y=1$, $z=-10$, $p\approx4.5\cdot10^{-5}$) $p(1-p)\approx0$: gradyan sıfıra yakın, model hatasından öğrenmez. Çapraz entropide aynı durumda $p-y\approx-1$: en büyük düzeltme. Ayrıca MSE$\circ\sigma$, $w$'ye göre dışbükey değildir (yerel minimumlar, düz platolar); çapraz entropi dışbükeydir (Hessian $\frac1nX^\top\mathrm{diag}(p(1-p))X\succeq0$).

Aşağıda kayıpları marjın fonksiyonu olarak karşılaştır ve büyük logitlerde naif hesabın nasıl bozulduğunu gör.

::lab l2

### Karar sınırı ve geometri {#sinir}

$p>\frac12\iff z>0\iff w^\top x+b>0$. **Karar sınırı** $w^\top x+b=0$ bir hiperdüzlemdir (2-B'de doğru); $w$ ona diktir (M0.1: iç çarpımın işareti, yarım düzlem). Lojistik regresyon bu yüzden **doğrusal bir sınıflandırıcıdır**: doğrusal olarak ayrılamayan sınıfları (XOR, iç içe halkalar) ayıramaz — ham özniteliklerde. Öznitelik dönüşümü ($x_1x_2$, $x_1^2+x_2^2$) ya da öğrenilen öznitelikler (sinir ağları, M3) bu sınırı aşar.

**$\|w\|$ sınırın keskinliğidir.** $x$'in sınıra işaretli uzaklığı $d=\frac{w^\top x+b}{\|w\|}$; böylece $z=\|w\|\,d$. $\|w\|$ büyüdükçe aynı uzaklıkta olasılık 0 ya da 1'e daha hızlı gider: $p$'nin 0.1'den 0.9'a çıktığı bandın genişliği $\frac{2\ln9}{\|w\|}\approx\frac{4.39}{\|w\|}$.

:::uyarı Ayrılabilir veride ağırlıklar sonsuza gider
Veri doğrusal olarak **tam ayrılabiliyorsa**, ayıran bir $w$'yi 2, 10, 1000 ile çarpmak her örneğin marjını büyütür ve kaybı azaltır: kayıp ancak $\|w\|\to\infty$'da 0'a iner. Gradyan inişi durmadan $\|w\|$'yi büyütür (logaritmik hızla), olasılıklar 0/1'e yapışır — **aşırı güven**. MLE burada sonlu bir çözüme sahip değildir. Çare: L2 cezası (M1.3), erken durdurma. (İlginç bir yan sonuç: GD'nin yönü, en büyük marjlı ayırıcıya — SVM çözümüne — yakınsar; Soudry ve ark. 2018.)
:::

Aşağıda iki sınıfı birbirinden uzaklaştır (ayrılabilir hâle getir), cezayı kapat ve $\|w\|$'nin büyümesini izle; sonra kaybı MSE yap.

::lab l1

### Sayısal kararlılık: logit üzerinden çalış {#kararlilik}

Kaybı "önce $p=\sigma(z)$, sonra $\log p$" diye hesaplamak iki yerde bozulur:

- $z\ll0$: $e^{-z}$ taşar ($z<-88$ float32'de, $<-709$ float64'te), $p=0$ olur, $\log0=-\infty$.
- $z\gg0$ ve $y=0$: $p$ tam 1'e yuvarlanır ($z\gtrsim17$ float32'de), $1-p=0$, yine $\log0$.

Çözüm, logaritmayı sigmoid'in içine almaktır:
$$\log\sigma(z)=-\log(1+e^{-z})=-\mathrm{softplus}(-z),\qquad \log(1-\sigma(z))=-\mathrm{softplus}(z),$$
$$\ell=\mathrm{softplus}(z)-yz,\qquad \mathrm{softplus}(z)=\max(z,0)+\log\big(1+e^{-|z|}\big).$$
Son biçimde $e^{-|z|}\le1$: hiçbir zaman taşmaz; `log1p` küçük argümanlarda hassasiyeti korur. **Kararlı sigmoid** için de işarete göre iki biçim: $z\ge0$ ise $\frac1{1+e^{-z}}$, değilse $\frac{e^z}{1+e^z}$ (modülün **b** aşaması).

PyTorch'ta bu yüzden: modelin son katmanı **sigmoid uygulamaz**, logit döndürür; kayıp `nn.BCEWithLogitsLoss` (logit alır, içeride kararlı formülü kullanır). `nn.Sigmoid` + `nn.BCELoss` hem kararsızdır hem de geri geçişte gereksiz $\sigma'$ çarpımı yapar. Olasılık gerektiğinde (çıkarımda) `torch.sigmoid(logit)` ayrıca çağrılır. Dengesiz sınıflar için `pos_weight` argümanı pozitif örneklerin kaybını ağırlıklandırır.

### Çok sınıfa ilk adım: bire karşı hepsi {#ovr}

$K$ sınıf için en basit yol: her sınıf için ayrı bir ikili sınıflandırıcı ("bu sınıf mı, değil mi?"), tahmin = en yüksek olasılıklı sınıf. Ağırlıklar $W\in\mathbb R^{d\times K}$ olarak tek tensörde tutulursa $K$ model aynı anda eğitilir (modülün **e** aşaması). Sorunları: olasılıkların toplamı 1 değildir ve sınıflar birbirinden habersiz eğitilir. M1.5'teki **softmax** bunları tek bir tutarlı olasılık modelinde çözer.

### Değerlendirme ve eşik {#degerlendirme}

$p>0.5$ eşiği, iki hatanın maliyeti eşitse doğrudur. Gerçekte yanlış negatif (kanseri kaçırmak) yanlış pozitiften çok pahalı olabilir: eşiği düşürmek duyarlılığı artırır, kesinliği azaltır (M0.4'teki Bayes ızgarası).

| ölçü | formül | ne söyler |
|---|---|---|
| doğruluk | (DP + DN) / hepsi | dengesiz sınıflarda yanıltıcı |
| kesinlik (precision) | DP / (DP + YP) | "pozitif" dediklerimin ne kadarı doğru |
| duyarlılık (recall) | DP / (DP + YN) | gerçek pozitiflerin ne kadarını yakaladım |
| F1 | kesinlik ve duyarlılığın harmonik ortalaması | tek sayılık özet |
| ROC-AUC | eşik taranınca DP oranı – YP oranı eğrisinin altındaki alan | eşikten bağımsız sıralama kalitesi |

**Kalibrasyon:** "%70 eminim" dediği örneklerin gerçekten ~%70'i pozitif mi? Lojistik regresyon (doğru belirlenmişse) genelde iyi kalibredir, çünkü doğrudan log-olabilirliği optimize eder; derin ağlar ise çoğu zaman aşırı güvenlidir (M3.9).

### Özet {#ozet}

| | doğrusal regresyon | lojistik regresyon |
|---|---|---|
| varsayım | $y\mid x\sim\mathcal N(w^\top x+b,\sigma^2)$ | $y\mid x\sim\text{Bernoulli}(\sigma(w^\top x+b))$ |
| kayıp | MSE | ikili çapraz entropi |
| $\partial\ell/\partial z$ | $2(\hat y-y)$ | $p-y$ |
| kapalı form | var | yok (dışbükey, GD / Newton) |
| PyTorch | `nn.MSELoss` | `nn.BCEWithLogitsLoss` (logit alır) |

**Akılda kalsın:** logit = log-odds, doğrusal. Kayıp NLL'dir ve gradyanı "tahmin − hedef". MSE'yi sigmoid'le birleştirme. Ayrılabilir veride ağırlıklar ceza olmadan sonsuza gider. Kaybı her zaman logit üzerinden, kararlı biçimde hesapla.

**Kaynaklar:** Stanford CS229 ders notları (lojistik regresyon, GLM'ler); CS231n, "Linear classification"; d2l.ai Bölüm 4; Murphy, *Probabilistic Machine Learning: An Introduction*, Bölüm 10; Soudry ve ark. (2018), "The implicit bias of gradient descent on separable data"; PyTorch `BCEWithLogitsLoss` belgesi.
