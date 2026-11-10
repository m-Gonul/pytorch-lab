### Zaman nereye gidiyor? {#zaman}

Horace He'nin "Making Deep Learning Go Brrrr" yazısındaki ayrımla, bir derin öğrenme programının süresi üç kaynaktan gelir:
1. **Hesap**: kayan nokta işlemleri (FLOP). Matris çarpımları burada yaşar ve donanımın tensör çekirdeklerini doldurur.
2. **Bellek trafiği**: verinin GPU belleği (HBM) ile hesap birimleri arasında taşınması. Eleman bazlı işlemler (toplama, aktivasyon, normalizasyon) neredeyse tamamen buradadır.
3. **Ek yük** (overhead): Python yorumlayıcısı, PyTorch'un işlem dağıtımı (dispatch), çekirdek başlatma. Küçük tensörlerde baskındır.

Eager modda (varsayılan PyTorch) her işlem ayrı bir çekirdektir: girdisini bellekten okur, çıktısını belleğe yazar. Derleyicinin ana işi, bu üç kaynaktan **ikinci ve üçüncüsünü** azaltmaktır.

### Roofline: bellek mi, hesap mı? {#roofline}

Bir işlemin **aritmetik yoğunluğu**, taşınan bayt başına yapılan FLOP'tur: $I=\text{FLOP}/\text{bayt}$. Roofline modeli (Williams, Waterman ve Patterson 2009) ulaşılabilir hızı şöyle sınırlar:
$$\text{hız}\le\min\big(\text{tepe FLOP/s},\ I\times\text{bant genişliği}\big).$$
İki sınırın kesiştiği **dirsek** noktası $I^*=\text{tepe}/\text{bant}$'tır. A100'de (bf16 yoğun ≈ 312 TFLOP/s, HBM ≈ 2 TB/s) $I^*\approx156$ FLOP/bayt; H100'de (≈ 989 TFLOP/s, 3.35 TB/s) ≈ 295.
- Eleman bazlı bir işlem fp32'de eleman başına 1 FLOP yapar ve 8 bayt taşır: $I=0.125$. Dirseğin binlerce kat altındadır, yani **bellek bağımlıdır**. Hesap birimleri zamanın çoğunda boş bekler.
- $n\times n$ matris çarpımı $2n^3$ FLOP yapar ve $3n^2$ eleman taşır: $I=2n/(3b)$ ($b$ = eleman başına bayt). bf16'da $n=4096$ için 1365: **hesap bağımlıdır**.

Bir bellek bağımlı işlemi hızlandırmanın tek yolu **daha az bayt taşımaktır**. Daha hızlı aritmetik hiçbir şey kazandırmaz. Donanım nesilleri boyunca FLOP/s, bant genişliğinden hızlı arttığı için dirsek sürekli sağa kayar ve daha çok işlem bellek bağımlı hâle gelir.

::lab l1

### Operatör füzyonu {#fuzyon}

Füzyon, ardışık eleman bazlı işlemleri tek bir çekirdekte birleştirir. Ara sonuçlar belleğe yazılmaz, yazmaçlarda kalır.

<figure class="fig wide"><svg viewBox="0 0 640 224" role="img" aria-label="Füzyonsuz üç çekirdek 24 MB, tek füzyonlu çekirdek 8 MB bellek trafiği yapar"><text x="100" y="46" text-anchor="end" class="sm">füzyonsuz</text><rect x="110" y="24" width="56.0" height="32" class="soft-a" stroke="var(--line)"/><rect x="166.0" y="24" width="56.0" height="32" class="soft-b" stroke="var(--line)"/><text x="166.0" y="72" text-anchor="middle" class="sm">mul</text><rect x="228" y="24" width="56.0" height="32" class="soft-a" stroke="var(--line)"/><rect x="284.0" y="24" width="56.0" height="32" class="soft-b" stroke="var(--line)"/><text x="284.0" y="72" text-anchor="middle" class="sm">add</text><rect x="346" y="24" width="56.0" height="32" class="soft-a" stroke="var(--line)"/><rect x="402.0" y="24" width="56.0" height="32" class="soft-b" stroke="var(--line)"/><text x="402.0" y="72" text-anchor="middle" class="sm">gelu</text><text x="470" y="46" class="ink">24 MB</text><text x="100" y="110" text-anchor="end" class="sm">füzyonlu</text><rect x="110" y="88" width="56.0" height="32" class="soft-a" stroke="var(--line)"/><rect x="166.0" y="88" width="56.0" height="32" class="soft-b" stroke="var(--line)"/><text x="166.0" y="136" text-anchor="middle" class="sm">mul + add + gelu</text><text x="234" y="110" class="ink">8 MB</text><rect x="110" y="160" width="14" height="12" class="soft-a" stroke="var(--line)"/><text x="130" y="171" class="sm">okunan (4 MB)</text><rect x="250" y="160" width="14" height="12" class="soft-b" stroke="var(--line)"/><text x="270" y="171" class="sm">yazılan (4 MB)</text><text x="10" y="198" class="sm">y = gelu(x·a + b), x: 10⁶ fp32. Her ayrı çekirdek x boyunda bir tensörü okur ve yazar: 3 · 8 MB = 24 MB. Füzyonlu çekirdek</text><text x="10" y="214" class="sm">x’i bir kez okur, y’yi bir kez yazar: 8 MB. FLOP aynı; bellek bağımlı işlemde hız kazancının üst sınırı 24 / 8 = 3.</text></svg><figcaption>Şekil 1 — Modülün el hesabı. Ara sonuçlar füzyonla belleğe hiç yazılmaz, yazmaçlarda kalır.</figcaption></figure>

**El hesabı.** $y=\mathrm{gelu}(x\cdot a+b)$, $x$: $10^6$ fp32 eleman ($a$, $b$ skaler, trafiğe katkısı yok). Füzyonsuz üç çekirdek (mul, add, gelu) her biri $N$ okur, $N$ yazar: $3\cdot2\cdot4N=\mathbf{24\,000\,000}$ bayt. Tek füzyonlu çekirdek $x$'i bir kez okur, $y$'yi bir kez yazar: $\mathbf{8\,000\,000}$ bayt. FLOP aynıdır; bellek bağımlı olduğu için beklenen hız kazancının üst sınırı $\mathbf3$'tür. Üstüne iki çekirdek başlatma ek yükü de kalkar.

**Füzyonun sınırları.** Hangi düğümlerin birleşebileceği grafın yapısına bağlıdır.
- **Tek tüketici**: bir ara değeri birden çok düğüm kullanıyorsa ve bunlar ayrı çekirdeklerdeyse değer belleğe yazılmak zorundadır. Aşama c bu basit kuralı kullanıyor. Gerçek derleyiciler (Inductor) aynı girdiyi kullanan tüketicileri tek çekirdekte toplar (yatay füzyon) ya da ucuz bir ara değeri gerektiği yerde **yeniden hesaplar**.
- **İndirgemeler** (sum, max, softmax'ın paydası) bir senkronizasyon noktasıdır. Eleman bazlı üreticiler indirgemenin çekirdeğine katılabilir (giriş tarafı füzyonu). İndirgemenin sonucunu kullanan sonraki işlemler ise genellikle yeni bir çekirdek ister (satır küçükse "persistent reduction" ile aynı çekirdekte kalabilir).
- **Çıktılar**: kullanıcıya dönen ya da geri geçiş için saklanan değerler (M6.1) her durumda yazılır.
- **Matris çarpımı**: kendisi hesap bağımlıdır; kazanç, ardından gelen eleman bazlı işlemleri (bias, aktivasyon) çarpımın çekirdeğinin sonuna eklemekten (epilog füzyonu) gelir.

::lab l2

### Grafı yakalamak: izleme ve torch.fx {#fx}

Füzyon için derleyicinin önce programın **grafını** görmesi gerekir. Eager modda graf hiçbir zaman bütün olarak var olmaz; işlemler tek tek çalışır. `torch.fx` **sembolik izleme** ile grafı yakalar. Fonksiyon gerçek tensörler yerine **Proxy** nesneleriyle çağrılır. Bir Proxy üzerindeki her işlem (`+`, `*`, `torch.tanh`, bir modül çağrısı) hesap yapmaz; graf'a bir **düğüm** kaydeder ve yeni bir Proxy döndürür. Aşama b bunu saf Python'da yazıyor.

FX grafında altı düğüm türü vardır: `placeholder` (girdi), `get_attr` (parametre/tampon), `call_function` (`torch.add`), `call_method` (`x.view(...)`), `call_module` (`self.conv(x)`), `output`. Graf düzenlenebilir bir veri yapısıdır; üzerinde **geçişler** (pass) yazılır.
- **Ortak alt ifade eleme** (CSE): aynı işlem aynı argümanlara iki kez uygulanıyorsa tek düğümde birleştir (değişmeli işlemlerde argüman sırası önemsiz).
- **Ölü kod eleme** (DCE): çıktıya ulaşmayan düğümleri sil.
- **Sabit katlama**: girdisi sabit olan alt grafları önceden hesapla.
- **Desen değiştirme**: bir alt grafı eşdeğer ve daha hızlı bir alt grafla değiştir. Aşağıdaki Conv + BN katlaması bunun klasik örneğidir.

Sembolik izlemenin sınırı: Proxy'nin **değeri** yoktur. `if x.sum() > 0` gibi veriye bağlı Python kontrol akışı izlenemez (Proxy'yi bool'a çeviremez), döngü sayısı veri bağımlı olamaz. torch.fx bu durumlarda hata verir. torch.compile'ın Dynamo katmanı tam bu sorunu çözmek için yazıldı.

### Conv + BatchNorm katlama {#katlama}

Değerlendirme modunda BN sabit istatistiklerle bir afin dönüşümdür; hemen önündeki konvolüsyona (ya da doğrusal katmana) katlanabilir. Böylece bir çekirdek ve bir tam okuma/yazma kalkar.

:::türetim Katlanmış ağırlık ve bias
Konvolüsyon $z=Wx+b$, BN (eval) $y=\gamma\frac{z-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ (çıkış kanalı başına). $s=\gamma/\sqrt{\sigma^2+\epsilon}$ diyelim:
$$y=s(Wx+b-\mu)+\beta=\underbrace{(sW)}_{W'}x+\underbrace{(b-\mu)s+\beta}_{b'} .$$
$s$ çıkış kanalı başına bir sayıdır; $W'$, $W$'nin her çıkış kanalı diliminin $s$ ile çarpılmasıdır. El hesabı: $w=2$, $b=1$, $\mu=3$, $\sigma^2=4$, $\gamma=0.5$, $\beta=1$, $\epsilon=0$: $s=0.25$, $w'=\mathbf{0.5}$, $b'=(1-3)\cdot0.25+1=\mathbf{0.5}$. Doğrulama: $x=1.7$ için $\mathrm{BN}(\mathrm{conv}(x))=1.35=0.5\cdot1.7+0.5$ (PyTorch'ta ölçüldü).
:::

Katlama **yalnız eval modunda** geçerlidir: eğitimde BN batch istatistiklerini kullanır ve onlar her batch'te değişir. Aşama d bunu FX ile yapıyor: `symbolic_trace`, BN düğümünün girdisi bir Conv2d düğümüyse ve conv çıktısının başka kullanıcısı yoksa conv modülünü katlanmışla değiştir, BN düğümünün kullanımlarını conv'a yönlendir ve BN düğümünü sil.

### torch.compile'ın katmanları {#compile}

PyTorch 2 (Ansel ve ark. 2024) eager modun esnekliğini koruyarak grafı yakalayan ve derleyen bir yığın getirdi.

<figure class="fig wide"><svg viewBox="0 0 640 220" role="img" aria-label="torch.compile yığını: Dynamo Python bytecode'undan FX grafı ve guard'lar çıkarır, AOTAutograd geri geçişi önceden izler, Inductor füzyonlu Triton ya da C++ çekirdekleri üretir; .item() gibi işlemler grafı ikiye böler"><defs><marker id="k1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="8" y="20" width="117" height="46" rx="5" class="soft-a"/><text x="66.5" y="38" text-anchor="middle" class="sm">Python fonksiyonu</text><text x="66.5" y="54" text-anchor="middle" class="sm">model(x)</text><rect x="135" y="20" width="117" height="46" rx="5" class="soft-p"/><text x="193.5" y="38" text-anchor="middle" class="sm">Dynamo</text><text x="193.5" y="54" text-anchor="middle" class="sm">bytecode → FX + guard</text><path d="M125,43 L134,43" class="ink" marker-end="url(#k1)"/><rect x="262" y="20" width="117" height="46" rx="5" class="soft-p"/><text x="320.5" y="38" text-anchor="middle" class="sm">AOTAutograd</text><text x="320.5" y="54" text-anchor="middle" class="sm">ileri + geri graf</text><path d="M252,43 L261,43" class="ink" marker-end="url(#k1)"/><rect x="389" y="20" width="117" height="46" rx="5" class="soft-p"/><text x="447.5" y="38" text-anchor="middle" class="sm">Inductor</text><text x="447.5" y="54" text-anchor="middle" class="sm">füzyon, döşeme</text><path d="M379,43 L388,43" class="ink" marker-end="url(#k1)"/><rect x="516" y="20" width="117" height="46" rx="5" class="soft-b"/><text x="574.5" y="38" text-anchor="middle" class="sm">çekirdekler</text><text x="574.5" y="54" text-anchor="middle" class="sm">Triton / C++</text><path d="M506,43 L515,43" class="ink" marker-end="url(#k1)"/><text x="8" y="96" class="ta">Graph break</text><rect x="8" y="106" width="150" height="40" rx="5" class="soft-p"/><text x="83.0" y="130.0" text-anchor="middle" class="sm">graf 1: out = x * 2</text><rect x="178" y="106" width="170" height="40" rx="5" class="soft-a"/><text x="263.0" y="124" text-anchor="middle" class="sm">Python: out.sum().item() > 0</text><text x="263.0" y="140" text-anchor="middle" class="sm">ve if dalı</text><rect x="368" y="106" width="150" height="40" rx="5" class="soft-p"/><text x="443.0" y="124" text-anchor="middle" class="sm">graf 2: tanh ya da sigmoid</text><text x="443.0" y="140" text-anchor="middle" class="sm">+ 1</text><path d="M158,126 L176,126" class="ink" marker-end="url(#k1)"/><path d="M348,126 L366,126" class="ink" marker-end="url(#k1)"/><text x="8" y="170" class="sm">.item() değeri izleme anında bilinmez: Dynamo grafı keser, Python’a döner, yeni graf başlatır (ölçüm: 1 break, 2 graf).</text><text x="8" y="186" class="sm">torch.where(out.sum() > 0, out.tanh(), out.sigmoid()) + 1 ile iki dal da tensör işlemi olur ve tek graf kalır.</text><text x="8" y="210" class="sm">Guard: grafın geçerli olduğu koşullar (şekil, dtype, cihaz, Python değerleri). Bozulursa yeniden derlenir.</text></svg><figcaption>Şekil 2 — torch.compile'ın katmanları (Ansel ve ark. 2024) ve aşama e'deki fonksiyonda bir graph break. Her graf ayrı derlenir; aradaki Python eager çalışır.</figcaption></figure>

- **Dynamo**, Python'un çerçeve değerlendirme API'sine (PEP 523) bağlanır ve fonksiyonun **bytecode**'unu çalıştırmadan önce analiz eder. PyTorch işlemlerini bir FX grafına toplar, desteklemediği bir şeyle karşılaşınca grafı keser (**graph break**), o kısmı normal Python'a bırakır ve sonra yeni bir graf başlatır. Her grafa **guard**'lar eklenir: grafın hangi koşullarda geçerli olduğu (girdi şekilleri, dtype, cihaz, kullanılan Python değerleri). Bir guard bozulursa yeniden derleme olur.
- **AOTAutograd** ileri grafın geri geçişini derleme anında izler ve ileri + geri ortak grafı kurar. Grafı ileri ve geri parçalara bölerken neyin saklanıp neyin yeniden hesaplanacağına bir min-cut algoritmasıyla karar verir. Ucuz eleman bazlı ara değerleri saklamak yerine yeniden hesaplamak hem belleği hem trafiği azaltır (M6.4'ün otomatik hâli).
- **Inductor** grafı alt düzey işlemlere indirger, füzyon gruplarını belirler ve GPU için **Triton**, CPU için C++/OpenMP çekirdekleri üretir. Matris çarpımları için cuBLAS gibi kütüphaneleri ya da (`max-autotune` ile) ayarlanmış Triton şablonlarını seçer.

Makale, 180'den fazla gerçek modelde A100 üzerinde Inductor'ın geometrik ortalama olarak çıkarımda 2.27 kat, eğitimde 1.41 kat hızlandırdığını bildirir. Kazanç modele bağlıdır: büyük matris çarpımlarının baskın olduğu modellerde az, eleman bazlı işlemlerin ve küçük tensörlerin çok olduğu modellerde çoktur.

**Modlar.** `torch.compile(model)` varsayılan; `mode="reduce-overhead"` CUDA graph'larıyla çekirdek başlatma ek yükünü kaldırır (küçük batch, çıkarım); `mode="max-autotune"` çekirdek seçeneklerini deneyerek seçer (uzun derleme süresi). `fullgraph=True` graph break olursa hata verir; `dynamic=True` şekil değişikliklerinde yeniden derlemeyi azaltmak için sembolik şekiller kullanır.

### Graph break'ler {#break}

Bir graph break performansı iki yoldan düşürür: grafın iki parçası ayrı ayrı derlenir (aralarında füzyon olamaz) ve aradaki Python eager çalışır. Sık nedenler:
- **Veriye bağlı kontrol akışı**: `if t.sum().item() > 0:`. `.item()` değeri izleme anında bilinmez.
- `print`, kayıt tutma, desteklenmeyen Python yapıları ya da kütüphaneler (NumPy çağrıları, bazı C uzantıları).
- Veri bağımlı çıktı şekilleri (`torch.nonzero`, boolean maskeyle indeksleme).

**Teşhis:** `torch._dynamo.explain(fn)(*args)` graf sayısını, break sayısını ve nedenlerini verir; `TORCH_LOGS="graph_breaks"` ortam değişkeni de her break'i yazdırır. Aşama e'deki fonksiyon için ölçtük: 1 break, 2 graf, neden `Tensor.item()`.

**Düzeltme:** kontrol akışını tensör işlemine çevirmek. `torch.where(out.sum() > 0, out.tanh(), out.sigmoid())` iki dalı da hesaplar ve eleman bazlı seçer; graf tek parça kalır. İki dal da ucuz eleman bazlı işlemler olduğu için fazladan hesap, kazanılan füzyonun yanında önemsizdir. Pahalı dallar için `torch.cond` (kontrol akışı operatörü) vardır.

:::uyarı Ölçerken dikkat
İlk çağrı derleme süresini içerir (saniyeler, büyük modelde dakikalar). Ölçümden önce ısınma çağrıları yapılmalı. GPU çekirdekleri eşzamansız çalıştığı için zaman ölçümünden önce `torch.cuda.synchronize()` çağrılmalı (M0.5'teki ölçüm gürültüsü deneyi). Girdi şekilleri her çağrıda değişiyorsa yeniden derlemeler kazancı yiyebilir; `torch._dynamo.config.cache_size_limit` aşılınca fonksiyon eager'a düşer.
:::

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| derlemek | `model = torch.compile(model)`; `mode="reduce-overhead"` / `"max-autotune"` |
| break'te hata | `torch.compile(fn, fullgraph=True)` |
| break analizi | `torch._dynamo.explain(fn)(*args)`; `TORCH_LOGS="graph_breaks,recompiles"` |
| üretilen kod | `TORCH_LOGS="output_code"` |
| FX grafı | `gm = torch.fx.symbolic_trace(model)`; `gm.graph.print_tabular()` |
| graf düzenlemek | `node.replace_all_uses_with(other)`; `gm.graph.erase_node(node)`; `gm.graph.lint(); gm.recompile()` |
| dal yerine | `torch.where(cond, a, b)`; pahalı dallar için `torch.cond` |
| Conv + BN katlama | `torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)` |

### Özet {#ozet}

- Süre = hesap + bellek trafiği + ek yük. Eleman bazlı işlemler bellek bağımlıdır ($I\approx0.1$–$1$ FLOP/bayt), dirsek A100'de ≈ 156.
- Füzyon ara değerleri belleğe yazmaz: $\mathrm{gelu}(xa+b)$ için 24 MB → 8 MB, kazanç üst sınırı 3. Sınırlar: çok tüketicili değerler, indirgemeler, çıktılar.
- torch.fx: Proxy ile sembolik izleme, altı düğüm türü, geçişler (CSE, DCE, desen değiştirme); veriye bağlı kontrol akışını izleyemez.
- Conv + BN (eval): $W'=sW$, $b'=(b-\mu)s+\beta$, $s=\gamma/\sqrt{\sigma^2+\epsilon}$ (el hesabı 0.5, 0.5).
- torch.compile: Dynamo (bytecode → FX + guard, graph break), AOTAutograd (ileri + geri, min-cut ile yeniden hesap), Inductor (füzyon, Triton/C++).
- Graph break'i `explain` ile bul, `torch.where`/`torch.cond` ile gider; ölçerken ısınma ve senkronizasyon.
