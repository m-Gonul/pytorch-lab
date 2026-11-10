### Ne zaman kendi türevini yazarsın? {#neden}

Autograd, PyTorch işlemlerinden kurulan her fonksiyonun türevini kendisi bulur (M6.1). Kendi geri geçişini yazmak şu durumlarda gerekir:
- **Autograd'ın göremediği işlemler.** NumPy, SciPy ya da harici bir kütüphane çağrısı, kendi yazdığın bir CUDA/Triton çekirdeği. Graf bu işlemleri kaydedemez.
- **Sayısal kararlılık.** Bileşenlerin türevleri ayrı ayrı taşan ama birleşik türevi zararsız olan işlemler (log-sum-exp, softmax + çapraz entropi).
- **Bilerek farklı bir gradyan.** Türevi sıfır ya da tanımsız olan işlemlere yararlı bir sinyal vermek (straight-through estimator), gradyanı ters çevirmek (alan uyarlaması), gradyanı kırpmak ya da ölçeklemek.
- **Bellek ve hız.** Daha az şey saklayıp geri geçişte yeniden hesaplamak; birkaç işlemi tek bir ileri ve tek bir geri çekirdekte birleştirmek.

Önce şunu sormak gerekir: aynı şey `torch` işlemleriyle yazılabiliyor mu? Yazılabiliyorsa autograd'ın türevi neredeyse her zaman doğru, test edilmiş ve `torch.compile` ile uyumludur. Özel Function, bir sorunu çözdüğü kanıtlandığında kullanılmalıdır.

### Sözleşme: forward, backward, ctx {#sozlesme}

`torch.autograd.Function`'dan türeyen bir sınıf iki statik yöntem tanımlar ve `MyFn.apply(...)` ile çağrılır.

<figure class="fig wide"><svg viewBox="0 0 640 240" role="img" aria-label="autograd.Function sözleşmesi: forward ctx'e tensörleri save_for_backward ile, diğer değerleri öznitelik olarak kaydeder; backward her girdi için bir gradyan döndürür; apply grafa tek bir düğüm ekler"><defs><marker id="c1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><text x="10" y="18" class="ta">ileri geçiş</text><text x="330" y="18" class="tb">geri geçiş</text><rect x="10" y="28" width="290" height="98" rx="5" class="soft-a"/><text x="20" y="46" class="sm">@staticmethod</text><text x="20" y="62" class="sm">def forward(ctx, x, lam):</text><text x="20" y="78" class="sm">    y = ...            # autograd kapalı</text><text x="20" y="94" class="sm">    ctx.save_for_backward(x)   # tensörler</text><text x="20" y="110" class="sm">    ctx.lam = lam              # diğerleri</text><rect x="330" y="28" width="300" height="98" rx="5" class="soft-b"/><text x="340" y="46" class="sm">@staticmethod</text><text x="340" y="62" class="sm">def backward(ctx, g):</text><text x="340" y="78" class="sm">    x, = ctx.saved_tensors   # sürüm denetimli</text><text x="340" y="94" class="sm">    gx = ... g ...  if ctx.needs_input_grad[0]</text><text x="340" y="110" class="sm">    return gx, None   # girdi başına bir değer</text><path d="M300,77 L326,77" class="ink" marker-end="url(#c1)"/><rect x="10" y="150" width="140" height="40" rx="5" class="soft-a"/><text x="20" y="168" class="sm">x (requires_grad)</text><rect x="190" y="150" width="160" height="40" rx="5" class="soft-p"/><text x="200" y="168" class="sm">MyFnBackward</text><rect x="390" y="150" width="120" height="40" rx="5" class="soft-a"/><text x="400" y="168" class="sm">y = MyFn.apply(x)</text><path d="M150,170 L186,170" class="ink" marker-end="url(#c1)"/><path d="M350,170 L386,170" class="ink" marker-end="url(#c1)"/><text x="10" y="214" class="sm">apply: forward’u no_grad altında çalıştırır, çıktıya grad_fn olarak tek bir düğüm bağlar. Motor geri geçişte bu düğüme</text><text x="10" y="230" class="sm">gelince backward’ı çağırır; dönen değerler forward’ın girdileriyle sırayla eşleşir (tensör olmayanlar için None).</text></svg><figcaption>Şekil 1 — Özel bir Function'ın iki yarısı. İleri geçişin içindeki işlemler grafa kaydedilmez; graf yalnız tek bir düğüm görür ve onun türevi, yazılan backward'dır.</figcaption></figure>

- **`forward(ctx, *girdiler)`** çıktıyı hesaplar. İçindeki işlemler grafa **kaydedilmez** (`apply` onu autograd kapalıyken çalıştırır); graf yalnız tek bir düğüm görür.
- **`ctx.save_for_backward(*tensörler)`** geri geçişte gereken tensörleri saklar. Tensörler `ctx.x = x` diye saklanmamalıdır. `save_for_backward` sürüm sayacı denetimini yapar (M6.1) ve düğüm ile tensör arasındaki referans döngüsünü önleyerek bellek sızıntısını engeller. Tensör olmayan değerler (sayılar, şekiller, bayraklar) ctx'e öznitelik olarak yazılır.
- **`backward(ctx, *grad_çıktılar)`** her çıktının gradyanını alır ve **her girdi için bir** değer döndürür. Gradyanı olmayan girdiler (tensör olmayanlar, tam sayı tensörleri) için `None` döndürülür. `ctx.needs_input_grad[i]` o girdinin gradyanının istenip istenmediğini söyler; istenmeyen gradyanı hesaplamamak işten tasarruftur.
- **Ek araçlar:** forward bir girdiyi yerinde değiştiriyorsa `ctx.mark_dirty(x)`; türevlenemeyen çıktılar (indeksler, maskeler) için `ctx.mark_non_differentiable(y)`; backward'ın kendisi türevlenemiyorsa `@once_differentiable`.
- **Yeni stil (PyTorch 2.0+):** `forward(*girdiler)` ctx almaz, saklama işi ayrı bir `setup_context(ctx, inputs, output)` yönteminde yapılır. Bu biçim `torch.func` dönüşümleriyle (vmap, jacrev; M6.3) birlikte kullanılabilir.

:::uyarı backward'ın girdileri "grad_output"tur, türev değil
`backward`, çıktının kendisinin değil **kaybın** çıktıya göre gradyanını alır (VJP'deki $\bar{\mathbf y}$). Görevi $\bar{\mathbf x}=\mathbf J^\top\bar{\mathbf y}$'yi döndürmektir. Sık yapılan hata, $\bar{\mathbf y}$ ile çarpmayı unutup yalnız yerel türevi döndürmektir. Kayıp tam olarak bu çıktıysa ($\bar{\mathbf y}=1$) hata görünmez; zincirin ortasına konunca ortaya çıkar. gradcheck ve aşağıdaki yön testi bunu yakalar.
:::

### Log-sum-exp: kararlı ileri, basit geri {#lse}

$\mathrm{lse}(\mathbf x)=\log\sum_ie^{x_i}$ softmax'ın, log-olabilirliklerin ve CTC gibi dinamik programlamaların yapı taşıdır.

:::türetim Kararlı ileri geçiş ve gradyan
$e^{1000}$ float64'te bile taşar. $m=\max_ix_i$ çıkarılırsa
$$\mathrm{lse}(\mathbf x)=m+\log\sum_ie^{x_i-m},$$
ve toplamdaki her terim $\le1$, en az biri tam 1'dir: ne taşma ne de $\log0$ olur. Gradyan:
$$\frac{\partial\,\mathrm{lse}}{\partial x_j}=\frac{e^{x_j}}{\sum_ie^{x_i}}=e^{x_j-\mathrm{lse}(\mathbf x)}=\mathrm{softmax}(\mathbf x)_j .$$
Geri geçiş için **çıktıyı** (ya da çıktı ile girdiyi) saklamak yeter: $\bar{\mathbf x}=\bar y\,e^{\mathbf x-y}$. Bu ifade de kararlıdır çünkü $x_j-y\le0$. El hesabı: $\mathbf x=(1,2)$ için $\mathrm{lse}=2+\log(1+e^{-1})=\mathbf{2.3133}$, $\partial/\partial x_2=e^{2-2.3133}=\mathbf{0.7311}$.
:::

Autograd max'ı çıkaran sürümün de türevini doğru alır (max'tan geçen gradyan, toplamdan geçenle tam olarak birbirini götürür). Özel bir Function'ın faydası burada daha az işlem ve daha az saklanan tensördür; aşama d bunu `torch.autograd.gradcheck` ve $x=1000$ girdisiyle sınıyor.

### Birleşik softmax + çapraz entropi {#ce}

:::türetim Tek düğümde gradyan p − y
$\ell=-\log\mathrm{softmax}(\mathbf z)_y=-z_y+\mathrm{lse}(\mathbf z)$. Bir önceki türetimle
$$\frac{\partial\ell}{\partial z_j}=-\mathbb 1[j=y]+\mathrm{softmax}(\mathbf z)_j=p_j-\mathbb 1[j=y].$$
Batch ortalamasında $1/N$ ile çarpılır. Geri geçiş için yalnız $\mathbf p$ (ya da log-olasılıklar) ve etiketler gerekir.
:::

Ayrı işlemlerle (`softmax` → `log` → `nll_loss`) yazmak iki şeyi bozar.
- **Kararlılık.** Bir olasılık float'ta 0'a yuvarlanırsa log $-\infty$ olur: $\mathbf z=(1000,0)$, $y=1$ için naif yol `inf` döndürür, `F.cross_entropy` doğru değeri (1000) verir.
- **Bellek.** Ölçtük: 64 × 1000'lik logitlerde naif zincir 2·N·C float (512 000 bayt), `F.cross_entropy` N·C (256 000 bayt) saklar. Softmax çıktısı ve log çıktısı ayrı tensörlerdir ve ikisi de saklanır.

Aşama c birleşik işlemi NumPy'da yazıyor ve yalnız olasılıkları saklıyor. Büyük sözlüklü dil modellerinde (C = 50 000+) son katmanın logitleri en büyük aktivasyondur; birleşik ve parçalı (chunked) CE çekirdekleri bu yüzden ayrıca geliştirilir.

### Straight-through estimator {#ste}

Nicemleme (`round`), işaret fonksiyonu ya da örnekleme gibi işlemlerin türevi neredeyse her yerde **sıfırdır**, sıçrama noktalarında tanımsızdır. Bu işlemler ağın içindeyse ondan önceki parametrelere hiç gradyan ulaşmaz.

Bengio, Léonard ve Courville'in (2013) incelediği **straight-through estimator** (STE) basit bir hile önerir: ileri geçişte gerçek işlem (`round`), geri geçişte **özdeşlik** kullanılır, sanki işlem hiç yokmuş gibi gradyan aynen geçirilir. Bu bir türev değil, bir **tahmincidir**. Sezgisi şudur: nicemleme küçük bir gürültü gibi davranıyorsa ($q(w)\approx w$), sürekli değişkene göre gradyan iyi bir yön verir.

<figure class="fig wide"><svg viewBox="0 0 640 196" role="img" aria-label="İki kasıtlı gradyan: round ileri geçişte yuvarlar, gerçek türevi 0'dır, STE geri geçişte 1 kullanır; gradyan tersine çevirme ileri geçişte özdeşliktir, geri geçişte gradyanı eksi lambda ile çarpar"><text x="150" y="20" text-anchor="middle" class="sm">ileri</text><text x="300" y="20" text-anchor="middle" class="sm">gerçek türev</text><text x="460" y="20" text-anchor="middle" class="sm">tanımlanan geri</text><rect x="10" y="34" width="620" height="44" rx="5" class="soft-a"/><text x="20" y="61" class="ink">round(x)</text><text x="150" y="61" text-anchor="middle" class="sm">round(x)</text><text x="300" y="61" text-anchor="middle" class="sm">0 (gerçek)</text><text x="460" y="61" text-anchor="middle" class="tb">1 (STE)</text><rect x="10" y="88" width="620" height="44" rx="5" class="soft-p"/><text x="20" y="115" class="ink">GR(x)</text><text x="150" y="115" text-anchor="middle" class="sm">x</text><text x="300" y="115" text-anchor="middle" class="sm">1 (gerçek)</text><text x="460" y="115" text-anchor="middle" class="tb">−λ (ters çevrilmiş)</text><text x="10" y="166" class="sm">El hesabı: x = (0.3, 1.7), w = (2, 3), L = Σ round(xᵢ)·wᵢ = 0·2 + 2·3 = 6.</text><text x="10" y="184" class="sm">∂L/∂x₁: gerçek 0, STE ile w₁ = 2. Gradyan tersine çevirme (λ = 0.5) ile L = Σ GR(xᵢ)·wᵢ için −0.5·2 = −1.</text></svg><figcaption>Şekil 2 — Bilerek gerçek türevden farklı tanımlanan iki geri geçiş ve modülün el hesapları. İkisi de gradcheck'ten geçmez, geçmemelidir: türev değil, birer tahmin ya da hiledir.</figcaption></figure>

El hesabı: $\mathbf x=(0.3,1.7)$, $\mathbf w=(2,3)$, $L=\sum_i\mathrm{round}(x_i)w_i=0\cdot2+2\cdot3=\mathbf6$. Gerçek $\partial L/\partial x_1=\mathbf0$; STE ile $\mathbf{2}$ ($=w_1$).

**Kullanım yerleri.**
- **Nicemleme farkında eğitim** (QAT). Ağırlıklar tam hassasiyette saklanır, ileri geçişte nicemlenir, gradyan STE ile gizli ağırlığa ulaşır. Pratikte gradyan, nicemleme aralığının dışında sıfırlanır ("kırpmalı STE"); yoksa gizli ağırlık anlamsız bölgeye kaçar.
- **VQ-VAE.** Kodlayıcı çıktısı en yakın kod kitabı vektörüne yuvarlanır; kod çözücünün gradyanı kodlayıcıya STE ile aktarılır.
- **İkili ağlar** (BinaryConnect): `sign` ileri, kırpmalı özdeşlik geri.

PyTorch'ta STE için Function yazmak bile gerekmez: `x + (round(x) - x).detach()` ileri geçişte `round(x)`'i verir. Geri geçişte `detach` edilen kısım sabittir ve gradyan özdeşlikle geçer.

Aşağıdaki deneyde tek ağırlıklı bir modeli 1–6 bitle eğit. Gerçek türevle ağırlığın hiç hareket etmediğini, STE ile en yakın ızgara noktasına ulaştığını, hedef iki ızgara noktası arasındaysa ağırlığın eşiğin etrafında salındığını gör.

::lab l2

### Gradyanı tersine çevirmek {#grl}

Ganin ve Lempitsky (2015), **alan uyarlaması** için bir katman önerdi. İleri geçişte özdeşliktir, geri geçişte gradyanı $-\lambda$ ile çarpar. Bir öznitelik çıkarıcının üstünde iki baş vardır: etiket sınıflandırıcısı ve kaynak/hedef alanını ayırt eden bir sınıflandırıcı. Alan başının önüne gradyan tersine çevirme katmanı (GRL) konunca tek bir geri geçiş iki şeyi aynı anda yapar: alan başı alanları ayırmayı öğrenir, öznitelik çıkarıcı ise onu **kandırmayı** öğrenir, yani alandan bağımsız öznitelikler üretir. GAN'ın minimaks oyunu (M5.2) burada iki ayrı optimizasyon adımı yerine tek bir işaret değişimiyle kurulur. El hesabı: $\lambda=0.5$, $L=\sum_i\mathrm{GR}(x_i)w_i$ için $\partial L/\partial x_1=-0.5\cdot2=\mathbf{-1}$.

`apply(x, lam)`'de `lam` bir sayıdır; backward onun için `None` döndürür. Yaygın uygulamalar (ve aşama d) ileri geçişte `x.view_as(x)` döndürür. PyTorch 2.x'te girdiyi aynen döndürmek de çalışıyor: autograd çıktıyı girdinin takma adı olan yeni bir tensöre sarıp düğümü ona bağlıyor (2.15'te ölçüldü, iki biçim de aynı gradyanı verdi). `view_as` alışkanlığı zararsızdır ve niyeti açık eder.

### gradcheck: sonlu farklarla doğrulama {#gradcheck}

Özel bir backward yazıldığında ilk test, sayısal türevle karşılaştırmaktır. `torch.autograd.gradcheck(fn, girdiler)` her girdi bileşeni için merkezi farkı hesaplar ve analitik Jacobian'la karşılaştırır: $|\text{sayısal}-\text{analitik}|\le\text{atol}+\text{rtol}\cdot|\text{analitik}|$ (varsayılanlar eps = 10⁻⁶, atol = 10⁻⁵, rtol = 10⁻³).

:::türetim Neden float64? Hatanın iki bileşeni
Merkezi fark $D_h=\frac{f(x+h)-f(x-h)}{2h}$. Taylor açılımıyla **kesme hatası** $\frac{h^2}6|f'''|$'tür. Bilgisayarda $f$ ancak $\pm u|f|$ hassasiyetle hesaplanır ($u$: birim yuvarlama); iki değerin farkı $2h$'ye bölündüğünde **yuvarlama hatası** $\approx u|f|/h$ olur. Toplam
$$E(h)\approx\frac{h^2}6|f'''|+\frac{u|f|}h,\qquad h^*=\Big(\frac{3u|f|}{|f'''|}\Big)^{1/3}\sim u^{1/3},\qquad E(h^*)\sim u^{2/3}.$$
- float64: $u=2^{-53}\approx1.1\cdot10^{-16}$, $h^*\approx5\cdot10^{-6}$, $E\sim10^{-11}$. Varsayılan eps = 10⁻⁶ bu optimuma yakındır.
- float32: $u=2^{-24}\approx6\cdot10^{-8}$. eps = 10⁻⁶'da yuvarlama hatası $u|f|/h\approx0.06|f|$'dir: **doğru** türev bile reddedilir. En iyi $h\approx4\cdot10^{-3}$'te bile hata $10^{-5}$ mertebesindedir.

İleri farkta kesme hatası $\frac h2|f''|$ olduğu için en iyi hata yalnız $\sim\sqrt u$'dur; merkezi fark tercih edilir.
:::

Aşağıdaki deney, float32 yuvarlamasını adım adım benzeterek hata–h eğrisini çiziyor. Sağ kol kesme hatası, sol kol yuvarlama hatasıdır. Ölçülen en iyi h kuramla tutuyor ($e^x$, $x_0=1$, float32: 0.0032'ye karşı 0.0039).

::lab l1

**Maliyet ve yön testi.** Tam gradcheck girdi boyutu $d$ kadar ileri geçiş ister; büyük girdilerde pahalıdır. Ucuz bir alternatif rastgele yönlerde karşılaştırmaktır: $\frac{f(\mathbf x+\epsilon\mathbf v)-f(\mathbf x-\epsilon\mathbf v)}{2\epsilon}\approx\langle\nabla f,\mathbf v\rangle$. $k$ yön $O(k)$ geçiş ister ve rastgele bir yön, gradyandaki bir hatayı olasılık 1 ile görür (hatalı bileşene dik olmadıkça). Aşama c bunu yazıyor. `gradgradcheck` aynı testi ikinci türevler için yapar.

:::uyarı STE ve GRL gradcheck'ten geçmez, geçmemelidir
Bu iki işlemde backward bilerek gerçek türevden farklıdır. gradcheck bunlarda kalır; bu bir hata değil, tanımdır. Test, tahmincinin kendisine değil, onun kullanıldığı **geri kalan** kodun doğruluğuna yapılır.
:::

### Bellek dostu SiLU {#silu}

$\mathrm{silu}(x)=x\,\sigma(x)$ (Swish). Türevi $\sigma(x)+x\,\sigma(x)(1-\sigma(x))=\sigma(x)\big(1+x(1-\sigma(x))\big)$.

Naif `x * torch.sigmoid(x)` yazılırsa sigmoid çıktısını, çarpma da iki girdisini (x ve sigmoid çıktısı) saklar. Ölçüm: 3 referans, ama **2 ayrı depo** (x ve σ(x); çarpmanın sakladığı σ(x) sigmoid'in sakladığıyla aynı tensördür). Özel bir Function yalnız $x$'i saklar ve $\sigma(x)$'i geri geçişte yeniden hesaplar: bellek yarıya iner, karşılığında geri geçişte bir sigmoid daha hesaplanır. Bu, M6.4'teki aktivasyon checkpointing'in tek işlemlik hâlidir. PyTorch'un yerleşik `F.silu`'su zaten böyle yapar (ölçüm: 1 depo).

:::uyarı Kaynak notu: aşama e'nin "üç tensör" ifadesi
Aşama e naif SiLU'nun "üç tensör sakladığını" söylüyor. Bu, saklanan **referans** sayısı olarak doğrudur. Bellek açısından ise iki depo vardır, çünkü sigmoid'in sakladığı çıktı ile çarpmanın sakladığı ikinci girdi aynı tensördür (M6.1'deki "aynı depo bir kez sayılır" kuralı). Kazanç bu yüzden üçte bir değil yarıdır.
:::

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| Function iskeleti | `class F(torch.autograd.Function): @staticmethod def forward(ctx, x): ...; @staticmethod def backward(ctx, g): ...` |
| saklama | `ctx.save_for_backward(x, y)`; geri: `x, y = ctx.saved_tensors` |
| tensör olmayan | `ctx.lam = lam`; backward'da `return g * ..., None` |
| gereksizi atlamak | `if ctx.needs_input_grad[0]: ...` |
| STE (Function'sız) | `x + (torch.round(x) - x).detach()` |
| GRL | ileri `x.view_as(x)`, geri `-ctx.lam * g, None` |
| doğrulama | `torch.autograd.gradcheck(F.apply, (x.double().requires_grad_(),))` |
| ikinci türev | `torch.autograd.gradgradcheck(...)` |
| vmap uyumu | `setup_context` + `generate_vmap_rule = True` |

### Özet {#ozet}

- Özel Function: autograd'ın göremediği işlemler, kararlılık, kasıtlı gradyanlar, bellek ve hız için. Mümkünse önce torch işlemleriyle yaz.
- Sözleşme: `forward(ctx, …)` grafa tek düğüm ekler; tensörler `save_for_backward` ile, diğerleri ctx özniteliği olarak saklanır; `backward` $\bar{\mathbf y}$ alır, her girdi için bir değer (ya da None) döndürür.
- lse: max çıkararak kararlı; gradyanı softmax (el hesabı 2.3133, 0.7311). Softmax + CE'nin gradyanı $\mathbf p-\mathbf y$; birleşik işlem kararlıdır ve yarı bellek saklar (ölçüm 256 000'e karşı 512 000 bayt).
- STE: ileri round, geri özdeşlik (el hesabı 6; 0 ve 2). Kırpmalı STE, QAT, VQ-VAE. GRL: ileri özdeşlik, geri $-\lambda$ (−1).
- gradcheck: merkezi fark, $E\sim h^2+u/h$, en iyi $h\sim u^{1/3}$; float32'de eps = 10⁻⁶ doğru türevi bile reddeder, float64 kullan. Yön testi $O(k)$.
- Bellek dostu SiLU: yalnız $x$'i sakla, $\sigma$'yı yeniden hesapla (2 depo → 1).
