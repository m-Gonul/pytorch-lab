### Çok katmanlı algılayıcı (MLP) {#mlp}

Doğrusal modeller (M1) özniteliklerin doğrusal bir fonksiyonunu öğrenir; iki ay ya da XOR gibi problemleri çözemez. Çözüm, öznitelikleri de öğrenmektir: girdiyi önce bir dizi **gizli katmandan** geçirip yeni temsiller üret, sonra son katmanda bunlar üzerinde doğrusal bir model kur. Her gizli katman bir doğrusal dönüşüm ve ardından eleman eleman **doğrusal olmayan** bir aktivasyondur.

**Gösterim** (Nielsen ve ders slaytlarıyla aynı; $l=1..L$ katman):

| sembol | anlamı | shape |
|---|---|---|
| $w^l_{jk}$ | $(l-1)$. katmandaki $k$. nörondan $l$. katmandaki $j$. nörona ağırlık | $W^l$: $(n_l\times n_{l-1})$ |
| $b^l_j$ | $l$. katmandaki $j$. nöronun bias'ı | $b^l$: $(n_l)$ |
| $z^l=W^la^{l-1}+b^l$ | ağırlıklı girdi (ön-aktivasyon) | $(n_l)$ |
| $a^l=\sigma(z^l)$ | aktivasyon; $a^0=x$ | $(n_l)$ |

$w^l_{jk}$'deki indis sırası ("hedef, kaynak") ilk bakışta ters görünür ama matris biçimini $z^l=W^la^{l-1}+b^l$ yapar: $W^l$'nin $j$. satırı $j$. nöronun ağırlık vektörüdür (M0.2'deki satır görünümü). PyTorch'un `nn.Linear` ağırlığı da $(n_{\text{out}},n_{\text{in}})$'dir.

<figure class="fig wide"><svg viewBox="0 0 640 220" role="img" aria-label="Üç katmanlı bir MLP: girdi, iki gizli katman ve çıktı; ileri geçişte z ve a, geri geçişte delta"><g class="sm"><g class="ca-f"><circle cx="70" cy="70" r="12"/><circle cx="70" cy="150" r="12"/></g><g class="cp-f"><circle cx="230" cy="40" r="12"/><circle cx="230" cy="110" r="12"/><circle cx="230" cy="180" r="12"/><circle cx="390" cy="40" r="12"/><circle cx="390" cy="110" r="12"/><circle cx="390" cy="180" r="12"/></g><g class="cb-f"><circle cx="550" cy="110" r="12"/></g><g class="mute"><line x1="82" y1="70" x2="218" y2="40"/><line x1="82" y1="70" x2="218" y2="110"/><line x1="82" y1="70" x2="218" y2="180"/><line x1="82" y1="150" x2="218" y2="40"/><line x1="82" y1="150" x2="218" y2="110"/><line x1="82" y1="150" x2="218" y2="180"/><line x1="242" y1="40" x2="378" y2="40"/><line x1="242" y1="40" x2="378" y2="110"/><line x1="242" y1="40" x2="378" y2="180"/><line x1="242" y1="110" x2="378" y2="40"/><line x1="242" y1="110" x2="378" y2="110"/><line x1="242" y1="110" x2="378" y2="180"/><line x1="242" y1="180" x2="378" y2="40"/><line x1="242" y1="180" x2="378" y2="110"/><line x1="242" y1="180" x2="378" y2="180"/><line x1="402" y1="40" x2="538" y2="110"/><line x1="402" y1="110" x2="538" y2="110"/><line x1="402" y1="180" x2="538" y2="110"/></g><text x="70" y="205" text-anchor="middle">a⁰ = x</text><text x="230" y="210" text-anchor="middle">z¹ = W¹a⁰ + b¹, a¹ = σ(z¹)</text><text x="390" y="210" text-anchor="middle">a² = σ(W²a¹ + b²)</text><text x="550" y="150" text-anchor="middle">a³ (çıktı)</text><text x="150" y="20" text-anchor="middle">W¹: (3 × 2)</text><text x="310" y="20" text-anchor="middle">W²: (3 × 3)</text><text x="470" y="60" text-anchor="middle">W³: (1 × 3)</text><text x="590" y="50" class="tb">δ³ = ∂C/∂z³</text><text x="430" y="12" class="tb">δ² = (W³)ᵀδ³ ⊙ σ′(z²)</text></g></svg><figcaption>Şekil 1 — İleri geçiş soldan sağa: her katman z ve a üretir (ve saklar). Geri geçiş sağdan sola: çıkış hatası δᴸ'den başlayıp her katmanda Wᵀ ile geri taşınır ve σ′ ile ölçeklenir.</figcaption></figure>

**Neden doğrusal olmayan aktivasyon?** Aktivasyonsuz iki katman $W^2(W^1x+b^1)+b^2=(W^2W^1)x+(W^2b^1+b^2)$ tek bir doğrusal katmandır (M0.2: bileşke = çarpım). Derinlik, ancak aradaki doğrusal olmayanlıkla bir şey kazandırır. Evrensel yaklaşım teoremi (tek gizli katman, yeterli genişlik, sürekli fonksiyonlar) M3.8'de.

### Çıkış katmanı ve kayıp {#cikis}

Çıkış katmanının aktivasyonu ve kayıp, **neyi tahmin ettiğine** göre seçilir (M0.4: kayıp = NLL):

| görev | çıkış nöronu | çıkış aktivasyonu | kayıp |
|---|---|---|---|
| regresyon (tek/çok değer) | tahmin edilen değer başına 1 | yok (doğrusal); pozitif çıkış için softplus/ReLU, aralık için sigmoid/tanh ölçekli | MSE, MAE, Huber |
| ikili sınıflandırma | 1 | sigmoid (kayba gömülü: `BCEWithLogitsLoss`) | BCE |
| çok etiketli (her sınıf bağımsız) | sınıf başına 1 | sınıf başına sigmoid | sınıf başına BCE |
| çok sınıflı (tek doğru sınıf) | sınıf başına 1 | softmax (kayba gömülü: `cross_entropy`) | CE |

### Geri yayılımın koşulları ve dört denklem {#bp}

Geri yayılımın uygulanabilmesi için maliyet $C$ iki koşulu sağlamalı:

1. **Örnek başına maliyetlerin ortalaması** olarak yazılabilmeli: $C=\frac1n\sum_xC_x$. Geri yayılım tek bir örneğin gradyanını hesaplar; toplam gradyan bunların ortalamasıdır (mini-batch'te batch ortalaması).
2. Ağın **çıktılarının fonksiyonu** olarak yazılabilmeli: $C_x=C(a^L(x),y(x))$. Özyineleme çıktı katmanından başlar.

Tek bir örnek için, $l$. katmandaki $j$. nöronun **hatasını** $\delta^l_j=\frac{\partial C}{\partial z^l_j}$ diye tanımlayalım: o nöronun ağırlıklı girdisindeki küçük bir değişikliğin maliyeti ne kadar değiştirdiği. ($a$'ya göre değil $z$'ye göre tanımlamak formülleri sadeleştirir.) $\odot$ **Hadamard** (eleman eleman) çarpımdır: $[1,2]\odot[3,4]=[3,8]$.

:::tanım Geri yayılımın dört denklemi
$$\textbf{BP1}\quad\delta^L=\nabla_{a^L}C\odot\sigma'(z^L)\qquad\qquad\textbf{BP2}\quad\delta^l=\big((W^{l+1})^\top\delta^{l+1}\big)\odot\sigma'(z^l)$$
$$\textbf{BP3}\quad\frac{\partial C}{\partial b^l_j}=\delta^l_j\qquad\qquad\textbf{BP4}\quad\frac{\partial C}{\partial w^l_{jk}}=a^{l-1}_k\,\delta^l_j\quad\Big(\text{matris: }\frac{\partial C}{\partial W^l}=\delta^l(a^{l-1})^\top\Big)$$
:::

:::türetim BP1–BP4, zincir kuralından
**BP1.** $C$, $z^L_j$'ye yalnız $a^L_j=\sigma(z^L_j)$ üzerinden bağlıdır: $\delta^L_j=\frac{\partial C}{\partial a^L_j}\frac{\partial a^L_j}{\partial z^L_j}=\frac{\partial C}{\partial a^L_j}\sigma'(z^L_j)$.
**BP2.** $z^l_j$, sonraki katmanın **tüm** $z^{l+1}_n$'lerine $a^l_j$ üzerinden bağlıdır (çok değişkenli zincir kuralı, yollar toplanır): $\delta^l_j=\sum_n\frac{\partial C}{\partial z^{l+1}_n}\frac{\partial z^{l+1}_n}{\partial z^l_j}=\sum_n\delta^{l+1}_n\frac{\partial z^{l+1}_n}{\partial z^l_j}$. $z^{l+1}_n=\sum_iw^{l+1}_{ni}\sigma(z^l_i)+b^{l+1}_n$ toplamında $z^l_j$ yalnız $i=j$ teriminde geçer: $\frac{\partial z^{l+1}_n}{\partial z^l_j}=w^{l+1}_{nj}\sigma'(z^l_j)$. Böylece $\delta^l_j=\sum_nw^{l+1}_{nj}\delta^{l+1}_n\,\sigma'(z^l_j)$; $\sum_nw^{l+1}_{nj}\delta^{l+1}_n$, $(W^{l+1})^\top\delta^{l+1}$'in $j$. elemanıdır.
**BP3.** $\frac{\partial C}{\partial b^l_j}=\frac{\partial C}{\partial z^l_j}\frac{\partial z^l_j}{\partial b^l_j}=\delta^l_j\cdot1$.
**BP4.** $\frac{\partial C}{\partial w^l_{jk}}=\frac{\partial C}{\partial z^l_j}\frac{\partial z^l_j}{\partial w^l_{jk}}=\delta^l_ja^{l-1}_k$.
**Koşullar:** $\sigma$ türevlenebilir (ReLU'da $z=0$'da alt-gradyan, M0.3); maliyet yukarıdaki iki koşulu sağlar. **Not:** BP1–BP4, M0.3'teki ters mod otomatik türevin bu mimariye özel, elle açılmış hâlidir; M3.1'in motoru aynı sayıları üretir.
:::

**Algoritma:** (1) ileri geçiş: her katmanda $z^l$ ve $a^l$'yi hesapla ve **sakla**; (2) BP1 ile $\delta^L$; (3) BP2 ile $\delta^{L-1},\dots,\delta^1$; (4) BP3–BP4 ile gradyanlar. Mini-batch'te her örneğin gradyanları ortalanır ve
$$w^l_{jk}\leftarrow w^l_{jk}-\frac\eta B\sum_{x\in\mathcal B}a^{l-1}_k(x)\,\delta^l_j(x),\qquad b^l_j\leftarrow b^l_j-\frac\eta B\sum_{x\in\mathcal B}\delta^l_j(x).$$

**Maliyet:** geri geçiş, ileri geçişle aynı mertebede iş yapar (BP2'deki $W^\top\delta$ ve BP4'teki $\delta a^\top$ iki matris çarpımı; ileride bir tane): kabaca ileri geçişin **2 katı FLOP**. Bir eğitim adımı ≈ 3 ileri geçiş — tüm parametrelerin gradyanı için, parametre sayısından bağımsız (M0.3'teki ucuz gradyan ilkesi).

### Denklemlerin söyledikleri: yavaş öğrenme {#icgoruler}

Dört denklem yalnız bir hesap yöntemi değil, öğrenmenin **ne zaman yavaşladığını** da söyler:

1. **BP4: düşük aktivasyonlu nöronun ağırlıkları yavaş öğrenir.** $\frac{\partial C}{\partial w^l_{jk}}=a^{l-1}_k\delta^l_j$: girdi nöronu $a^{l-1}_k\approx0$ ise o bağlantının gradyanı küçüktür. (Bias'ın girdisi her zaman 1 olduğu için bu sorun bias'ta yoktur.) Sıfır merkezli olmayan aktivasyonların (sigmoid, ReLU) bir sonucu: bir katmanın tüm girdileri pozitifse, bir nöronun tüm ağırlıklarının gradyanı **aynı işaretli** olur (hepsi $\delta^l_j$'nin işaretini taşır) — ağırlıklar zikzak yaparak öğrenir (ders alıştırması 2).
2. **BP1: doymuş çıkış nöronu yavaş öğrenir.** Sigmoid çıkış $a^L\approx0$ ya da $1$ iken $\sigma'(z^L)\approx0$: kuadratik maliyetle hata büyük olsa bile $\delta^L\approx0$. Çapraz entropide $\sigma'$ sadeleşir (M1.4, aşağıdaki deney).
3. **BP2: doymuş gizli nöronlar da yavaş öğrenir.** $\delta^l_j$'de $\sigma'(z^l_j)$ çarpanı var: doymuş bir nörona giren tüm ağırlıklar yavaş öğrenir — $\sum_nw^{l+1}_{nj}\delta^{l+1}_n$ bunu telafi edecek kadar büyük değilse.
4. **Kararsız gradyanlar.** BP2'yi katmanlar boyunca açınca **çarpanların çarpımı** çıkar:
$$\delta^l=\mathrm{diag}(\sigma'(z^l))(W^{l+1})^\top\,\mathrm{diag}(\sigma'(z^{l+1}))(W^{l+2})^\top\cdots\delta^L.$$
 Her çarpan tipik olarak 1'den küçükse (sigmoid'de $\sigma'\le\frac14$) ilk katmanların gradyanı üstel olarak **söner**; büyükse **patlar**. Derin ağlarda ilk katmanlar çok yavaş öğrenir (Glorot ve Bengio 2010). Çareler M3.5 (başlatma, aktivasyon), M3.6 (normalizasyon) ve M4.8 (artık bağlantılar) konularıdır.

::lab l1

### Matris biçimi: batch ve modüler katmanlar {#matris}

Pratikte örnekler batch'te **satır** olarak gelir: $X\in\mathbb R^{B\times n_0}$. Bir doğrusal katman $Y=XW^\top+b$ (PyTorch) ya da $Y=XW+b$ (CS231n, $W$ $(n_{\text{in}}\times n_{\text{out}})$). Gelen gradyan $dY=\frac{\partial C}{\partial Y}$ (Y ile aynı shape) için, M0.2'deki $dA=GB^\top$, $dB=A^\top G$ kurallarından ($Y=XW+b$ biçiminde):
$$dX=dY\,W^\top,\qquad dW=X^\top dY,\qquad db=\mathbf1^\top dY\ (\text{satırlar üzerinden toplam}).$$
$dW=X^\top dY=\sum_{\text{örnekler}}x_i^\top dy_i$: BP4'ün tüm batch için toplamı (dış çarpımların toplamı, M0.2). $db$, yayılan bias'ın gradyanıdır: yayın, geri geçişte **toplamaya** dönüşür.

| katman | ileri | geri | sakla (cache) |
|---|---|---|---|
| doğrusal | $Y=XW+b$ | $dX=dYW^\top$, $dW=X^\top dY$, $db=\sum_idY_i$ | $X$ (ve $W$) |
| ReLU | $Y=\max(0,X)$ | $dX=dY\odot[X>0]$ | $X$ (ya da maske) |
| sigmoid | $Y=\sigma(X)$ | $dX=dY\odot Y(1-Y)$ | $Y$ |
| softmax + CE | $L=\frac1B\sum_i-\log P_{i,y_i}$ | $dS=\frac1B(P-Y_{\text{onehot}})$ | $P$ |

**Modüler tasarım** (CS231n): her katman `forward(x) -> (out, cache)` ve `backward(dout, cache) -> (dx, dparams)`. Ağın ileri geçişi bu fonksiyonları sırayla, geri geçişi ters sırayla çağırır. PyTorch'ta `torch.autograd.Function`'ın `forward(ctx, ...)` / `backward(ctx, grad)` ve `ctx.save_for_backward` yapısı tam olarak budur (M3.3).

:::uyarı Shape doğru, sonuç yanlış
Geri geçiş hatalarının çoğu shape'i bozmaz: ReLU maskesini unutmak, $W$ yerine $W^\top$ kullanmak (kare matrislerde), batch ortalamasını iki kez almak. Shape kontrolü yetmez; **gradyan kontrolü** gerekir (M0.3: merkezi fark, float64, göreli hata). Modülün **d** aşamasında elle yazdığın gradyanları autograd ile karşılaştıracaksın.
:::

Aşağıda bu denklemlerle eğitilen bir MLP'yi iki ay verisinde dene (M3.1'deki deneyle aynı motor): aktivasyonu sigmoid yap ve derinliği artır — üstteki içgörülerin etkisini gör.

::lab l2

### Birçok modeli birden: torch.func ile topluluk {#ensemble}

Aynı mimariden 16 bağımsız model (farklı başlatma) eğitip tahminlerini ortalamak, M2.5'teki bagging fikrinin sinir ağı karşılığıdır (derin topluluklar; belirsizlik kestiriminde güçlü bir yöntem, M3.9). 16 modeli döngüyle çalıştırmak 16 ayrı küçük kernel çağrısı demektir. `torch.func` ile parametreleri yığıp tek bir vektörleştirilmiş çağrı yapılır:

```python
from torch.func import stack_module_state, functional_call, vmap
import copy
models = [MLP() for _ in range(16)]
params, buffers = stack_module_state(models)          # her tensör: (16, *shape)
base = copy.deepcopy(models[0]).to("meta")            # yalnız yapı
f = lambda p, b, x: functional_call(base, (p, b), (x,))
out = vmap(f, in_dims=(0, 0, None))(params, buffers, X)   # (16, B, C), tek çağrı
```

Modülün **e** aşaması bunu kurar ve döngü sonucuyla karşılaştırır.

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| ileri | $z^l=W^la^{l-1}+b^l$, $a^l=\sigma(z^l)$; $z$ ve $a$'yı sakla |
| hata | $\delta^l=\partial C/\partial z^l$ |
| BP1 / BP2 | $\delta^L=\nabla_aC\odot\sigma'(z^L)$; $\delta^l=(W^{l+1})^\top\delta^{l+1}\odot\sigma'(z^l)$ |
| BP3 / BP4 | $\partial C/\partial b=\delta$; $\partial C/\partial W^l=\delta^l(a^{l-1})^\top$ |
| batch | $dX=dYW^\top$, $dW=X^\top dY$, $db=\sum dY$ |
| yavaş öğrenme | düşük $a$ (BP4), doymuş çıkış (BP1), doymuş gizli nöron (BP2), çarpanlar zinciri (sönen/patlayan gradyan) |
| maliyet | geri ≈ 2 × ileri; ileri geçiş ara değerleri bellekte |

**Kaynaklar:** Nielsen, *Neural Networks and Deep Learning*, Bölüm 2 (dört denklem ve türetimleri) ve Bölüm 3 (çapraz entropi ve yavaş öğrenme); Toygar, BİL 442 Bölüm 10–11 slaytları (koşullar, BP1–BP4, içgörüler, güncelleme adımı, MLP ile regresyon/sınıflandırma) ve Alıştırma 2; CS231n, "Neural Networks Part 1" ve Assignment 2 (modüler katmanlar); Rumelhart, Hinton, Williams (1986); Glorot ve Bengio (2010), "Understanding the difficulty of training deep feedforward neural networks"; Prince, *Understanding Deep Learning*, Bölüm 7; PyTorch "Model ensembling" eğitimi.
