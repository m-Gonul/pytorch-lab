### Çalışırken kurulan graf {#graf}

M3.3'te `loss.backward()`'ı bir kara kutu gibi kullandık. Bu modülde kutuyu açıyoruz. PyTorch'un autograd'ı **tanımla-ve-çalıştır** (define-by-run) türündedir: hesap grafı önceden bildirilmez, Python kodu çalıştıkça işlem işlem kurulur. Gradyan isteyen (`requires_grad=True`) bir tensörü kullanan her işlem, sonucun yanında bir **geri fonksiyonu** düğümü yaratır ve bunu çıktının `grad_fn` alanına yazar.

- `y.grad_fn` → `y`'yi üreten işlemin geri düğümü (ör. `MulBackward0`, `AddmmBackward0`).
- `y.grad_fn.next_functions` → girdilerin düğümleri: `(düğüm, çıktı_indeksi)` çiftleri. Graf geriye doğru bağlıdır; ileri yönde bağlantı tutulmaz.
- **Yaprak** tensörler (kullanıcının yarattığı, `requires_grad=True` olanlar: parametreler) bir `AccumulateGrad` düğümüyle temsil edilir. Bu düğümün tek işi gelen gradyanı `.grad`'a **eklemektir**.
- Gradyan istemeyen tensörlerin düğümü yoktur (`next_functions`'ta `None`). Bir işlemin girdilerinden biri bile gradyan istiyorsa çıktı da ister.

Grafın her yinelemede yeniden kurulması, Python'un `if`, `for` ve özyinelemesini doğrudan kullanmayı sağlar. Her yineleme farklı bir graf kurabilir; dinamik dizi uzunlukları ya da koşullu hesaplar için ek bir şey gerekmez. Bedeli, grafın her seferinde kurulup atılmasının küçük bir ek yüküdür (M6.5'te `torch.compile` bunu hedefler).

<figure class="fig wide"><svg viewBox="0 0 640 262" role="img" aria-label="Hesap grafı: a ve b yaprakları, c = ab, d = c + a, L = d kare; geri geçişte L'den 16, c'den a'ya 48 ve b'ye 32, a'ya doğrudan 16 gradyan akar"><defs><marker id="f" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="g" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><rect x="10" y="40" width="120" height="40" rx="5" class="soft-a"/><text x="70.0" y="56" text-anchor="middle" class="sm">a = 2 (yaprak)</text><text x="70.0" y="71" text-anchor="middle" class="sm">AccumulateGrad</text><rect x="10" y="170" width="120" height="40" rx="5" class="soft-a"/><text x="70.0" y="186" text-anchor="middle" class="sm">b = 3 (yaprak)</text><text x="70.0" y="201" text-anchor="middle" class="sm">AccumulateGrad</text><rect x="200" y="105" width="120" height="40" rx="5" class="soft-p"/><text x="260.0" y="121" text-anchor="middle" class="sm">c = ab = 6</text><text x="260.0" y="136" text-anchor="middle" class="sm">MulBackward0: a, b</text><rect x="360" y="40" width="120" height="40" rx="5" class="soft-p"/><text x="420.0" y="56" text-anchor="middle" class="sm">d = c + a = 8</text><text x="420.0" y="71" text-anchor="middle" class="sm">AddBackward0: —</text><rect x="510" y="40" width="120" height="40" rx="5" class="soft-p"/><text x="570.0" y="56" text-anchor="middle" class="sm">L = d² = 64</text><text x="570.0" y="71" text-anchor="middle" class="sm">PowBackward0: d</text><path d="M130,60 L196,115" class="ink" marker-end="url(#f)"/><path d="M130,190 L196,135" class="ink" marker-end="url(#f)"/><path d="M320,118 L356,72" class="ink" marker-end="url(#f)"/><path d="M130,52 L356,52" class="ink" marker-end="url(#f)"/><path d="M480,60 L506,60" class="ink" marker-end="url(#f)"/><path d="M510,92 L482,92" class="cb" marker-end="url(#g)"/><text x="495" y="108" text-anchor="middle" class="tb">16</text><path d="M380,84 L322,130" class="cb" marker-end="url(#g)"/><text x="368" y="118" class="tb">16</text><path d="M360,30 L134,30" class="cb" marker-end="url(#g)"/><text x="245" y="24" text-anchor="middle" class="tb">16 (doğrudan)</text><path d="M220,104 L132,84" class="cb" marker-end="url(#g)"/><text x="70" y="100" text-anchor="middle" class="tb">16·3 = 48</text><path d="M220,146 L132,166" class="cb" marker-end="url(#g)"/><text x="186" y="172" class="tb">16·2 = 32</text><text x="10" y="236" class="sm">Siyah: ileri geçiş (graf kurulurken). Turuncu: geri geçişte akan gradyanlar. a’nın düğümüne iki gradyan gelir:</text><text x="10" y="252" class="sm">motor bağımlılık sayısını 2 tutar, ikisi gelince toplar (48 + 16 = 64) ve .grad’a ekler. b.grad = 32.</text></svg><figcaption>Şekil 1 — El hesabındaki graf, PyTorch'un düğüm adlarıyla. Her düğümün altında sakladığı tensörler: MulBackward0 iki girdisini, PowBackward0 tabanını saklar, AddBackward0 hiçbir şey saklamaz (toplamanın türevi sabittir).</figcaption></figure>

### Ters mod türev: vektör–Jacobian çarpımı {#vjp}

Autograd'ın hesapladığı şey, her ara değişken $v$ için **eşlenik** (adjoint) $\bar v=\partial L/\partial v$'dir.

:::türetim Geri geçişin kuralı
$v_i$'yi kullanan düğümler (graftaki çocukları) $v_j=f_j(\dots,v_i,\dots)$ olsun. Çok değişkenli zincir kuralı, $v_i$'nin $L$'yi etkilediği tüm yolları toplar:
$$\bar v_i=\sum_{j\in\text{çocuk}(i)}\bar v_j\,\frac{\partial v_j}{\partial v_i}.$$
Vektör değişkenlerde her terim bir **vektör–Jacobian çarpımıdır** (VJP): $\bar{\mathbf v}_i\mathrel{+}=\mathbf J_{ij}^\top\bar{\mathbf v}_j$. İki sonuç doğar.
1. Her düğüm yalnız kendi VJP'sini bilmelidir; Jacobian'ı hiçbir zaman açıkça kurmak gerekmez. Örneğin $\mathbf Y=\mathbf A\mathbf B$ için $\bar{\mathbf A}=\bar{\mathbf Y}\mathbf B^\top$ ve $\bar{\mathbf B}=\mathbf A^\top\bar{\mathbf Y}$; Jacobian $(mn)\times(mk)$ boyutlu olurdu.
2. Bir değişkenin birden çok çocuğu varsa (dallanma) gradyanlar **toplanır**. Bir düğüm çalışmadan önce tüm çocuklarından gelen katkıların gelmesi gerekir; bu da bir topolojik sıra demektir.
:::

**El hesabı (Şekil 1).** $a=2$, $b=3$; $c=ab=6$, $d=c+a=8$, $L=d^2=64$. Geriye: $\bar d=2d=16$; Add bunu iki girdisine aynen iletir: $\bar c=16$ ve $a$'ya doğrudan 16. Mul: $\bar a\mathrel{+}=\bar c\cdot b=48$, $\bar b=\bar c\cdot a=\mathbf{32}$. Toplam $\bar a=48+16=\mathbf{64}$.

:::sezgi Neden ters mod?
$f:\mathbb R^n\to\mathbb R^m$ için ileri mod (M0.3, M6.3) her geçişte Jacobian'ın bir **sütununu** ($\mathbf J\mathbf u$), ters mod bir **satırını** ($\mathbf u^\top\mathbf J$) verir. Eğitimde $n$ milyonlarca parametre, $m=1$ (skaler kayıp): ters mod tüm gradyanı **tek** geri geçişle verir, ileri mod $n$ geçiş isterdi. Geri geçişin maliyeti ileri geçişin yaklaşık 2 katıdır (matris çarpımında iki VJP: $\bar{\mathbf A}$ ve $\bar{\mathbf B}$). Bedeli bellektir: VJP'ler ileri geçişteki değerlere ihtiyaç duyar, o değerler saklanmalıdır.
:::

### Motor: bağımlılık sayarak topolojik sıra {#motor}

`backward()` çağrıldığında C++'taki autograd motoru şunları yapar:
1. **Bağımlılıkları say.** Kökten (kaybın düğümü) başlayıp `next_functions` boyunca grafı gezer ve her düğüm için kaç kenarın ona geldiğini sayar. Şekil 1'de `AccumulateGrad(a)`'nın bağımlılığı 2'dir (Mul'dan ve Add'den).
2. **Hazır kuyruğu.** Kök, başlangıç gradyanı 1 ile kuyruğa girer. Bir düğüm çalışınca her girdisine bir gradyan gönderir; o girdinin girdi tamponunda gradyanlar **toplanır** ve bağımlılık sayısı bir azalır. Sayı sıfıra inen düğüm hazırdır ve kuyruğa girer (Kahn algoritması).
3. **Yapraklar.** `AccumulateGrad` gelen toplamı `.grad`'a ekler. `.grad` yoksa (`None`) gelen tensör (ya da kopyası) atanır.

Aşama b aynı motoru DFS ile kurulan topolojik sırayla yazıyor; iki yol da her düğümü, tüm tüketicileri işlendikten sonra çalıştırır. Gerçek motorda her cihaz (CPU, her GPU) için ayrı bir iş parçacığı vardır; GPU düğümleri o cihazın kuyruğunda çalışır.

**Ara tensörlerin `.grad`'ı.** Ara bir tensörün gradyanı hesaplanır (zincir kuralı onsuz ilerleyemez) ama **saklanmaz**: motor onu bir sonraki düğüme iletip atar. `.grad`'ı `None` kalır. İstenirse `t.retain_grad()` ya da bir hook kullanılır.

**Birikim.** `AccumulateGrad` toplar, yazmaz. `zero_grad` yapmadan iki kez `backward` çağırmak gradyanı ikiye katlar: el örneğinde `a.grad` $=2\cdot64=\mathbf{128}$. Bu bir hata değil, özelliktir: gradyan biriktirme (M6.4) tam olarak budur. PyTorch 2.0'dan beri `zero_grad()` gradyanları sıfırlamak yerine `None` yapar (`set_to_none=True`); bu hem bellek hem bir toplama işlemi kazandırır.

**`retain_graph`.** Geri geçiş, işi biten düğümlerin sakladığı tensörleri serbest bırakır. Aynı graf üzerinden ikinci kez geri geçiş yapmak için ilkinde `retain_graph=True` verilmelidir; yoksa "Trying to backward through the graph a second time" hatası alınır. `create_graph=True` ise geri geçişin kendisini de grafa kaydeder ve ikinci türevleri mümkün kılar (M6.3).

### Saklanan tensörler {#saklanan}

Bir düğümün geri geçişte neye ihtiyacı olduğu, türev formülünden okunur. PyTorch bunları ileri geçiş sırasında düğüme **saklanan tensörler** (saved tensors) olarak bağlar; Python'dan `grad_fn._saved_self`, `_saved_result` gibi adlarla görülebilir.

| işlem | türev | saklanan |
|---|---|---|
| $y=x_1x_2$ | $\bar x_1=\bar y\,x_2$, $\bar x_2=\bar y\,x_1$ | iki girdi |
| $y=x_1+x_2$ | $\bar x_i=\bar y$ | hiçbir şey |
| $y=e^x$ | $\bar x=\bar y\,y$ | **çıktı** |
| $y=\tanh x$ | $\bar x=\bar y(1-y^2)$ | çıktı |
| $y=\mathrm{ReLU}(x)$ | $\bar x=\bar y\,\mathbb 1[y>0]$ | çıktı |
| $y=\mathrm{GELU}(x)$ | $\bar x=\bar y\,\mathrm{GELU}'(x)$ | **girdi** |
| $\mathbf Y=\mathbf A\mathbf B$ | $\bar{\mathbf A}=\bar{\mathbf Y}\mathbf B^\top$, $\bar{\mathbf B}=\mathbf A^\top\bar{\mathbf Y}$ | iki girdi |
| $\mathrm{softmax}$ | $\bar{\mathbf x}=\mathbf y\odot(\bar{\mathbf y}-\langle\bar{\mathbf y},\mathbf y\rangle)$ | çıktı |
| LayerNorm | | girdi, satır başına ortalama ve $1/\sigma$ |

Tablo PyTorch 2.x'te `_saved_*` öznitelikleriyle doğrulandı. Bir tensörü "saklamak" kopyalamak değildir; düğüm aynı depoya bir referans tutar. Ama o depo, düğüm yaşadıkça serbest bırakılamaz. Eğitimdeki **aktivasyon belleği**, tam olarak geri geçiş bitene kadar canlı tutulan bu tensörlerdir.

**El hesabı.** `torch.tanh(a @ W)`, `a` (64, 128), `W` (128, 128), fp32: matmul `a` ve `W`'yi, tanh çıktısını saklar; **3** tensör, $4\,(64\cdot128+128\cdot128+64\cdot128)=\mathbf{131\,072}$ bayt. `W` bir parametredir ve zaten bellektedir; ek bellek yalnız `a` ve tanh çıktısıdır.

:::örnek Aynı depo bir kez sayılır: ReLU ve GELU
Pre-LN bir MLP bloğu: LN → Linear(d, 4d) → aktivasyon → Linear(4d, d) → artık toplama. ReLU kendi çıktısını saklar; ikinci Linear da girdisini, yani **aynı** tensörü saklar. Depo bir kez tutulur ve satır başına $d+d+4d=6d$ eleman kalır (LN girdisi, LN çıktısı, ReLU çıktısı). GELU ise **girdisini** saklar (türevi $x$'e bağlıdır); ikinci Linear GELU'nun çıktısını saklar. Bunlar iki ayrı tensördür: $10d$. Dropout eklenince çarpma düğümü maskeyi saklar ve ikinci Linear Dropout'un yeni çıktısını saklar; iki aktivasyonda da $14d$. Bu sayılar CPU'da `saved_tensors_hooks` ile depo adresleri sayılarak ölçüldü (6.0, 10.0, 14.0, 14.0).
:::

::lab l2

### Sürüm sayacı ve in-place işlemler {#surum}

Saklanan bir tensör ileri geçişten sonra yerinde (in-place) değiştirilirse geri geçiş yanlış değerle hesaplanır. PyTorch bunu sessizce yanlış gradyan üretmek yerine yakalar. Her tensörün bir **sürüm sayacı** vardır (`t._version`); her in-place işlem (`add_`, `mul_`, `x[i] = …`, `relu_`) onu bir artırır. Düğüm bir tensörü saklarken o anki sürümü de kaydeder; geri geçişte saklanan sürüm ile şimdiki sürümü karşılaştırır. Farklıysa şu hata gelir: "one of the variables needed for gradient computation has been modified by an inplace operation".

<figure class="fig wide"><svg viewBox="0 0 640 196" role="img" aria-label="Sürüm sayacı: exp çıktısını saklar, çıktı yerinde değiştirilince sürüm 0'dan 1'e çıkar ve geri geçiş hata verir; çarpma girdilerini sakladığı için çıktısını değiştirmek sorun olmaz"><defs><marker id="v" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="10" y="30" width="145" height="56" rx="5" class="soft-a"/><text x="82" y="52" text-anchor="middle" class="sm">x = randn(3)</text><text x="82" y="72" text-anchor="middle" class="sm">x: s0</text><rect x="167" y="30" width="145" height="56" rx="5" class="soft-p"/><text x="239" y="52" text-anchor="middle" class="sm">y = x.exp()</text><text x="239" y="72" text-anchor="middle" class="sm">ExpBackward0 saklar: (y, s0)</text><path d="M155,58 L165,58" class="ink" marker-end="url(#v)"/><rect x="324" y="30" width="145" height="56" rx="5" class="soft-b"/><text x="396" y="52" text-anchor="middle" class="sm">y.add_(1)</text><text x="396" y="72" text-anchor="middle" class="sm">y: s0 → s1</text><path d="M312,58 L322,58" class="ink" marker-end="url(#v)"/><rect x="481" y="30" width="145" height="56" rx="5" class="soft-b"/><text x="553" y="52" text-anchor="middle" class="sm">y.sum().backward()</text><text x="553" y="72" text-anchor="middle" class="sm">s1 ≠ s0 → RuntimeError</text><path d="M469,58 L479,58" class="ink" marker-end="url(#v)"/><text x="10" y="18" class="ta">exp çıktısını saklar: in-place değişiklik hata verir</text><rect x="10" y="128" width="145" height="56" rx="5" class="soft-p"/><text x="82" y="150" text-anchor="middle" class="sm">y = x * w</text><text x="82" y="170" text-anchor="middle" class="sm">MulBackward0 saklar: x, w</text><rect x="167" y="128" width="145" height="56" rx="5" class="soft-a"/><text x="239" y="150" text-anchor="middle" class="sm">y.add_(1)</text><text x="239" y="170" text-anchor="middle" class="sm">y: s0 → s1</text><path d="M155,156 L165,156" class="ink" marker-end="url(#v)"/><rect x="324" y="128" width="145" height="56" rx="5" class="soft-p"/><text x="396" y="150" text-anchor="middle" class="sm">y.sum().backward()</text><text x="396" y="170" text-anchor="middle" class="sm">y saklanmadı → sorun yok</text><path d="M312,156 L322,156" class="ink" marker-end="url(#v)"/><text x="10" y="116" class="ta">mul girdilerini saklar: çıktıyı değiştirmek serbest</text><text x="490" y="150" class="sm">Kural: bir tensörü</text><text x="490" y="166" class="sm">yerinde değiştirmek, onu</text><text x="490" y="182" class="sm">saklamış düğüm varsa hata.</text></svg><figcaption>Şekil 2 — Her tensörün bir sürüm sayacı vardır; her in-place işlem onu bir artırır. Düğüm bir tensörü saklarken sürümünü de not eder ve geri geçişte karşılaştırır. Hangi düğümün neyi sakladığı türev formülüne bağlıdır: exp'in türevi çıktısıdır, çarpmanınki öteki girdi.</figcaption></figure>

Kural kısadır: **in-place işlem, değiştirilen tensörü saklamış bir düğüm varsa hatadır**; yoksa serbesttir. Hangi düğümün neyi sakladığını yukarıdaki tablo söyler. Bu yüzden `nn.ReLU(inplace=True)` çoğu zaman güvenlidir: ReLU'nun girdisi genellikle bir Linear ya da BN çıktısıdır ve onu saklayan bir düğüm yoktur. Ama örneğin girdisi bir `exp` ya da `sigmoid` çıktısıysa hata verir.

İki ayrıntı:
- **Görünümler** (views: `x[0]`, `x.view(…)`, `x.T`) tabanlarıyla aynı sürüm sayacını paylaşır. Görünümü yerinde değiştirmek tabanı da "değiştirilmiş" yapar.
- **Yapraklar.** Gradyan isteyen bir yaprağı yerinde değiştirmek her zaman hatadır ("a leaf Variable that requires grad is being used in an in-place operation"). Parametre güncellemesi bu yüzden `torch.no_grad()` içinde yapılır; optimizasyon yöntemleri böyle çalışır.

### Hook'lar {#hook}

Autograd'a müdahale etmenin dört yolu vardır:
- **Tensor hook'u**: `t.register_hook(fn)`. Motor `t`'nin toplam gradyanını hesaplayınca `fn(grad)`'ı çağırır; `fn` yeni bir tensör döndürürse o kullanılır. Gradyan kırpma, gürültü ekleme ya da ara gradyanları kaydetme için kullanılır.
- **Biriktirme sonrası hook**: `p.register_post_accumulate_grad_hook(fn)`. Parametrenin `.grad`'ı güncellenince çağrılır; optimizasyon adımını geri geçişin içine almak (ve gradyanı hemen serbest bırakmak) için kullanılır.
- **Modül hook'ları**: `register_forward_pre_hook`, `register_forward_hook`, `register_full_backward_hook`. Aktivasyonları ya da modül gradyanlarını izlemek için kullanılır (M4.8'deki CAM, Grad-CAM).
- **Saklanan tensör hook'ları**: `torch.autograd.graph.saved_tensors_hooks(pack, unpack)`. Aşağıda.

**El hesabı: hook yalnız kendi yolunu etkiler.** $c$'ye gradyanı 2 ile çarpan bir hook koyalım. Hook $c$'nin **toplam** gradyanını değiştirir ($16\to32$), sonra bu değer girdilere dağılır: $\bar b=32\cdot2=\mathbf{64}$, $\bar a=32\cdot3+16=\mathbf{112}$. $a$'nın doğrudan yolu (Add'den gelen 16) hook'tan etkilenmez.

::lab l1

### Saklanan tensörleri paketlemek {#paketleme}

`saved_tensors_hooks(pack, unpack)` bağlamı içinde kurulan her düğüm, bir tensörü saklarken `pack(t)`'yi çağırır ve dönen nesneyi saklar. Geri geçişte `unpack(nesne)` tensörü geri verir. Bu, saklanan tensörlerin **nerede ve nasıl** tutulacağını kullanıcıya bırakır:
- **Sıkıştırma**: `pack` = `t.to(torch.bfloat16)`, `unpack` = özgün tipe dönüş. İleri geçiş tam hassasiyette kalır; yalnız geri geçişin kullandığı kopya yuvarlanır. Aktivasyon belleği yarıya iner, gradyanlar biraz yaklaşık olur (aşama e ölçüyor).
- **Boşaltma**: `torch.autograd.graph.save_on_cpu()` saklanan tensörleri CPU belleğine taşır ve geri geçişte GPU'ya geri getirir. GPU belleği yerine PCIe bant genişliği harcanır.
- **Yeniden hesaplama**: `pack` hiçbir şey saklamayıp `unpack` değeri yeniden hesaplarsa bu, aktivasyon checkpointing'in (M6.4) özüdür.

:::uyarı Ne zaman saklanır, ne zaman çağrılır?
`pack` ileri geçiş sırasında, düğüm **kurulurken** çağrılır; bağlamın etkin olması gereken an ileri geçiştir. Geri geçişi bağlamın içinde çağırmak bir şey değiştirmez. Ayrıca `pack` içinde tensörü yerinde değiştirmek yasaktır ve paketlenen değerin sürüm denetimi artık kullanıcının sorumluluğundadır.
:::

### Gradyan istememek: no_grad, inference_mode, detach {#nograd}

- `torch.no_grad()`: bağlam içinde graf kurulmaz, çıktılar `requires_grad=False` olur. Değerlendirme ve parametre güncellemesi için kullanılır.
- `torch.inference_mode()`: no_grad'a ek olarak sürüm sayacını ve görünüm takibini de kapatır; daha hızlıdır. Ama burada yaratılan tensörler daha sonra autograd'a giren bir hesapta kullanılamaz.
- `t.detach()`: aynı depoyu paylaşan, grafa bağlı olmayan bir tensör döndürür. Gradyanın bir yoldan akmasını kesmek için kullanılır (M5.5'teki taban çizgisi, hedef ağları). Depo paylaşıldığı için detach edilmiş tensörü yerinde değiştirmek, aslını saklamış bir düğüm varsa yine sürüm hatası verir.

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| grafı gezmek | `y.grad_fn`, `y.grad_fn.next_functions` |
| saklananları görmek | `y.grad_fn._saved_result`, `dir(y.grad_fn)` |
| ara gradyanı tutmak | `h.retain_grad()` |
| gradyanı değiştirmek | `h.register_hook(lambda g: g.clamp(-1, 1))` |
| sürümü görmek | `t._version` |
| saklananları sıkıştırmak | `with torch.autograd.graph.saved_tensors_hooks(pack, unpack): loss = model(x)` |
| CPU'ya boşaltmak | `with torch.autograd.graph.save_on_cpu(pin_memory=True): ...` |
| ikinci kez geri geçiş | `loss.backward(retain_graph=True)` |
| anomali ayıklama | `torch.autograd.set_detect_anomaly(True)` (hatayı üreten ileri işlemin yığın izini verir; yavaştır) |

### Özet {#ozet}

- Graf çalışırken kurulur: `grad_fn`, `next_functions`; yapraklar `AccumulateGrad`.
- Geri geçiş = eşleniklerin VJP'lerle yayılması; dallanmada toplanır (el örneği 64, 32). Ters mod skaler kayıpta tüm gradyanı tek geçişte verir.
- Motor bağımlılık sayar, hazır kuyruğuyla topolojik sırada çalışır; ara gradyanlar saklanmaz; `.grad` birikir (iki çağrı: 128).
- Düğümler türev formülünün gerektirdiğini saklar (mul girdileri, exp/tanh/ReLU çıktısı, GELU girdisi, matmul iki girdisi). Aktivasyon belleği budur (el örneği 3 tensör, 131 072 bayt). Paylaşılan depo bir kez sayılır: ReLU bloğu 6d, GELU 10d.
- Sürüm sayacı: saklanmış bir tensörü yerinde değiştirmek hata verir, saklanmamışı değiştirmek serbesttir.
- Hook'lar: tensor (yalnız o düğümün yolunu etkiler; 112, 64), biriktirme sonrası, modül, saklanan tensör (bf16'ya sıkıştırma, CPU'ya boşaltma, yeniden hesaplama).
