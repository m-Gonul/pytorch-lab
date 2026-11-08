### Problem: yumuşak kümeleme ve gizli değişkenler {#problem}

k-means (M2.7) her noktayı **kesin** olarak bir kümeye atar ve her kümeyi küresel varsayar. İki kümenin sınırındaki bir nokta için "%55 A, %45 B" demek daha dürüsttür; uzamış ya da farklı boyutlu kümeler kendi şekillerini hak eder. **Gauss karışım modeli (GMM)** bunları olasılıksal bir modelle çözer: veri, $K$ Gauss dağılımından birinden gelir ama **hangisinden geldiğini bilmiyoruz**.

Bu "gözlenmeyen değişken" fikri çok geneldir: gizli Markov modelleri (hangi durumdaydık?), konu modelleri (bu kelime hangi konudan?), eksik veri, VAE'ler ve difüzyon modelleri (M5: gizli kod $z$). Bu modülde hepsinin ortak aracı olan **beklenti–en büyükleme (EM)** algoritmasını en temiz örneğinde öğreniyoruz.

### Model {#model}

Üretim süreci (üretken model, M2.2):

1. Bir bileşen seç: $z\sim\text{Kategorik}(\pi_1,\dots,\pi_K)$, $\sum_k\pi_k=1$.
2. O bileşenden bir nokta üret: $x\mid z=k\sim\mathcal N(\mu_k,\Sigma_k)$.

$z$'yi bilmediğimiz için gözlenen verinin dağılımı tüm bileşenler üzerinden toplamdır (marjinal, M0.4):
$$p(x)=\sum_{k=1}^K\pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k).$$
Yeterli bileşenle bir GMM neredeyse her pürüzsüz yoğunluğu yaklaştırabilir: hem kümeleme hem **yoğunluk kestirimi** aracıdır.

**Sorumluluk (responsibility):** bir nokta gözlendiğinde hangi bileşenden geldiğinin sonsal olasılığı (Bayes kuralı):
$$r_{ik}=P(z_i=k\mid x_i)=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}.$$
$r_{i\cdot}$ bir olasılık vektörüdür: yumuşak atama.

### Neden doğrudan MLE zor? {#zor}

Log-olabilirlik
$$\ell(\theta)=\sum_{i=1}^n\log\sum_{k=1}^K\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)$$
**toplamın logaritmasını** içerir. Tek bir Gauss'ta log, üstel fonksiyonu yok eder ve kapalı form çözüm çıkar (M0.4: ortalama ve kovaryans); burada log toplamın dışında kaldığı için çıkmaz. Türevi sıfıra eşitlersek $\mu_k=\frac{\sum_ir_{ik}x_i}{\sum_ir_{ik}}$ buluruz — ama $r_{ik}$ de $\mu_k$'ya bağlıdır: bir sabit nokta denklemi. Bu denklem bize algoritmayı fısıldar: **$r$'yi sabit tut, parametreleri hesapla; parametreleri sabit tut, $r$'yi hesapla.**

### EM algoritması {#em}

$z$'yi bilseydik problem kolay olurdu: her bileşenin noktaları üzerinde ayrı ayrı MLE. EM, bilmediğimiz $z$'yi onun sonsal olasılıklarıyla (sorumluluklarla) "doldurur":

- **E adımı** (expectation): mevcut parametrelerle sorumlulukları hesapla, $r_{ik}$.
- **M adımı** (maximization): sorumlulukları **ağırlık** sayarak her bileşen için ağırlıklı MLE:
$$N_k=\sum_ir_{ik},\quad\pi_k=\frac{N_k}n,\quad\mu_k=\frac1{N_k}\sum_ir_{ik}x_i,\quad\Sigma_k=\frac1{N_k}\sum_ir_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top.$$

$N_k$ bileşenin "etkin nokta sayısı"dır. Her adımın neden olabilirliği hiç düşürmediği girişteki kutuda kanıtlandı: E adımı bir alt sınırı ($\mathcal F$, ELBO) olabilirliğe eşitler, M adımı alt sınırı büyütür.

<figure class="fig wide"><svg viewBox="0 0 640 195" role="img" aria-label="EM'in alt sınır yorumu: log-olabilirlik eğrisi, E adımında ona değen alt sınır, M adımında alt sınırın tepesine gidiş"><g class="sm"><line x1="40" y1="170" x2="600" y2="170" class="mute"/><text x="600" y="186" text-anchor="end">θ</text><polyline points="40,149.2 45,149.1 50,148.9 55,148.8 60,148.6 65,148.5 70,148.3 75,148.0 80,147.8 85,147.5 90,147.2 95,146.9 100,146.5 105,146.1 110,145.7 115,145.2 120,144.7 125,144.1 130,143.5 135,142.8 140,142.0 145,141.2 150,140.4 155,139.4 160,138.4 165,137.4 170,136.2 175,135.0 180,133.6 185,132.2 190,130.8 195,129.2 200,127.5 205,125.8 210,123.9 215,122.0 220,119.9 225,117.8 230,115.6 235,113.3 240,110.9 245,108.4 250,105.9 255,103.2 260,100.5 265,97.7 270,94.9 275,92.0 280,89.1 285,86.1 290,83.1 295,80.1 300,77.1 305,74.1 310,71.1 315,68.1 320,65.1 325,62.2 330,59.3 335,56.5 340,53.8 345,51.2 350,48.7 355,46.3 360,44.1 365,41.9 370,39.9 375,38.1 380,36.5 385,35.0 390,33.7 395,32.6 400,31.6 405,30.9 410,30.4 415,30.1 420,30.0 425,30.1 430,30.4 435,30.9 440,31.6 445,32.6 450,33.7 455,35.0 460,36.5 465,38.1 470,39.9 475,41.9 480,44.1 485,46.3 490,48.7 495,51.2 500,53.8 505,56.5 510,59.3 515,62.2 520,65.1 525,68.1 530,71.1 535,74.1 540,77.1 545,80.1 550,83.1 555,86.1 560,89.1 565,92.0 570,94.9 575,97.7 580,100.5 585,103.2 590,105.9 595,108.4 600,110.9" class="ca w3" fill="none"/><text x="440" y="24" class="ta">log p(x | θ)</text><polyline points="135,166.7 140,162.5 145,158.4 150,154.6 155,151.0 160,147.6 165,144.4 170,141.4 175,138.6 180,136.0 185,133.6 190,131.3 195,129.3 200,127.5 205,125.9 210,124.5 215,123.3 220,122.3 225,121.5 230,120.8 235,120.4 240,120.2 245,120.2 250,120.4 255,120.8 260,121.4 265,122.2 270,123.2 275,124.3 280,125.7 285,127.3 290,129.1 295,131.1 300,133.3 305,135.7 310,138.3 315,141.1 320,144.0 325,147.2 330,150.6 335,154.2 340,158.0 345,162.0 350,166.2" class="cb w2" fill="none"/><text x="96" y="209" class="tb">alt sınır 𝓕(q, θ)</text><line x1="200" y1="170" x2="200" y2="127.5" class="mute dash"/><circle cx="200" cy="127.5" r="4" class="ink-f"/><text x="200" y="186" text-anchor="middle">θₜ</text><text x="192" y="118" text-anchor="end">E: 𝓕 = log p (KL = 0)</text><line x1="242.8" y1="170" x2="242.8" y2="109.5" class="cp dash"/><circle cx="242.8" cy="120.2" r="4" class="cp-f"/><circle cx="242.8" cy="109.5" r="4" class="ca-f"/><text x="249" y="186" class="tp">θₜ₊₁</text><text x="251" y="124" class="tp">M: 𝓕'nin tepesi</text><text x="251" y="104" class="ta">log p(θₜ₊₁) ≥ 𝓕(θₜ₊₁) ≥ 𝓕(θₜ) = log p(θₜ)</text></g></svg><figcaption>Şekil 1 — E adımı, olabilirliğe θₜ'de dokunan bir alt sınır kurar; M adımı o alt sınırın en yüksek noktasına gider. Olabilirlik alt sınırın üstünde olduğu için yeni nokta en az eski kadar iyidir. Tekrar edince olabilirlik bir yerel tepeye tırmanır.</figcaption></figure>

Aşağıda EM'i iterasyon iterasyon izle: elipslerin verinin şekline nasıl oturduğunu, sınırdaki belirsiz noktaları ve log-olabilirliğin hiç düşmeden yükseldiğini gör.

::lab l1

### Sayısal uygulama {#sayisal}

- **Log uzayı:** $\mathcal N(x\mid\mu,\Sigma)$ yüksek boyutta ya da uzak noktalarda alttan taşar. Her şeyi log'da hesapla: $\log r_{ik}=\log\pi_k+\log\mathcal N_{ik}-\mathrm{LSE}_j(\log\pi_j+\log\mathcal N_{ij})$ (M1.5'in log-sum-exp hilesi).
- **Log-yoğunluk:** $\log\mathcal N(x\mid\mu,\Sigma)=-\frac d2\log2\pi-\frac12\log\det\Sigma-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)$. $\Sigma^{-1}$'i açıkça alma: Cholesky $\Sigma=LL^\top$, $\log\det\Sigma=2\sum\log L_{ii}$, Mahalanobis terimi $\|L^{-1}(x-\mu)\|^2$ (üçgen çözüm). Tüm $(n,K)$ log-yoğunlukları einsum ile tek seferde (modülün **c** aşaması).
- **Tekillik:** bir bileşen tek bir noktaya çökerse $\Sigma_k\to0$ ve olabilirlik $\to\infty$ (MLE'nin patolojisi). Çare: kovaryansa küçük bir taban ($\Sigma_k+\epsilon I$, scikit-learn `reg_covar`) ya da kovaryansa önsel (MAP-EM).
- **Kovaryans türleri:** tam ($d(d+1)/2$ parametre/bileşen), köşegen ($d$), küresel (1), ortak (bileşenler aynı $\Sigma$'yı paylaşır). Az veride ya da yüksek boyutta kısıtlı türler aşırı uyumu önler.

### k-means, EM'in sert limitidir {#kmeans}

Tüm bileşenler aynı küresel kovaryansa sahip olsun, $\Sigma_k=\sigma^2I$, ve $\pi_k$ eşit. Sorumluluk
$$r_{ik}\propto\exp\Big(-\frac{\|x_i-\mu_k\|^2}{2\sigma^2}\Big).$$
$\sigma\to0$'da en yakın merkezin terimi diğerlerine üstel olarak baskın gelir: $r_{ik}\to1$ en yakın $k$ için, 0 diğerleri için — **k-means'in atama adımı**. M adımı ($r$ 0/1 iken) kümelerin ortalaması — **güncelleme adımı**. k-means, "sert EM"dir. GMM'nin kazandırdıkları: yumuşak atamalar, kümeye özgü şekil (kovaryans) ve boyut ($\pi_k$), olasılıksal bir model (yoğunluk, olabilirlik, model seçimi).

### Bileşen sayısı: BIC {#bic}

Olabilirlik $K$ arttıkça (en iyi çözümde) artar; tek başına $K$ seçemez. **Bayesçi bilgi ölçütü:**
$$\mathrm{BIC}=-2\ell(\hat\theta)+p\log n,$$
$p$ serbest parametre sayısı (2B tam kovaryansta bileşen başına $2+3$, artı $K-1$ karışım ağırlığı). En küçük BIC'li model seçilir; $p\log n$ cezası fazla bileşeni cezalandırır. AIC ($-2\ell+2p$) daha az cezalandırır, daha çok bileşen seçmeye eğilimlidir. Doğrulama kümesindeki log-olabilirlik de doğrudan kullanılabilir.

### EM mi gradyan mı? {#gradyan}

Aynı log-olabilirliği gradyanla da büyütebiliriz (modülün **d** aşaması: `torch.distributions.MixtureSameFamily` + Adam). Karşılaştırma:

| | EM | gradyan inişi |
|---|---|---|
| adım boyu | yok (kapalı form M adımı) | öğrenme oranı seçilmeli |
| kısıtlar ($\pi$ toplamı 1, $\Sigma\succ0$) | kendiliğinden sağlanır | yeniden parametrelendirme gerekir (softmax, Cholesky) |
| monotonluk | garanti | yok |
| genellik | M adımı kapalı formsa | her türevlenebilir model |
| ölçek | tam veri geçişleri | mini-batch ile çok büyük veri |

Derin üretken modellerde M adımı kapalı form değildir; EM'in yapısı korunup M adımı yerine gradyan adımları atılır (genelleştirilmiş EM) ya da E adımı bir sinir ağıyla yaklaşık yapılır (VAE'nin kodlayıcısı, amortize çıkarım — M5.1). Girişteki $\log p=\mathcal F+\mathrm{KL}$ ayrışımı VAE'nin ELBO'suyla **aynı** denklemdir.

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| GMM | $p(x)=\sum_k\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)$; gizli bileşen $z$ |
| sorumluluk | $r_{ik}=P(z=k\mid x_i)$, Bayes kuralı; yumuşak atama |
| E adımı | $r$'yi hesapla (log uzayında, LSE ile) |
| M adımı | ağırlıklı MLE: $\pi_k=N_k/n$, ağırlıklı ortalama ve kovaryans |
| garanti | olabilirlik hiç düşmez ($\log p=\mathcal F+\mathrm{KL}$); yerel optimum |
| tuzaklar | çöken bileşen (kovaryansa taban), başlatma (birden çok deneme, k-means ile başlat) |
| k-means | eşit küresel kovaryans, $\sigma\to0$ limiti |

**Kaynaklar:** Stanford CS229 ders notları, "Mixtures of Gaussians and the EM algorithm"; Bishop, *Pattern Recognition and Machine Learning* §9.2–9.4 (EM'in genel biçimi, k-means ilişkisi, alt sınır şekli); Dempster, Laird, Rubin (1977), "Maximum likelihood from incomplete data via the EM algorithm"; Neal ve Hinton (1998), "A view of the EM algorithm that justifies incremental, sparse, and other variants"; Murphy, *PML1* §8.7; scikit-learn `GaussianMixture` belgesi (kovaryans türleri, `reg_covar`, BIC).
