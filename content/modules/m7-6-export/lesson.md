### Python'dan çıkmak {#neden}

Eğitimde Python'un esnekliği bir avantajdır (M6.1'deki tanımla-ve-çalıştır graf). Dağıtımda ise bir yüktür. Bir sunucu, telefon ya da tarayıcı Python yorumlayıcısı çalıştırmak istemeyebilir; her çağrıda yorumlayıcı ek yükü ödenir; model kodu ve bağımlılıkları sürümlenip taşınmalıdır. **Dışa aktarma** (export), modeli Python'dan bağımsız, tam ve sabit bir hesap grafına dönüştürür. Bu graf saklanabilir, başka bir çalışma zamanında yüklenebilir ve o çalışma zamanının derleyicisiyle optimize edilebilir.

<figure class="fig wide"><svg viewBox="0 0 640 206" role="img" aria-label="Dağıtım yolları: Python modeli torch.export ile ExportedProgram'a, oradan AOTInductor paketine, ONNX'e ya da ExecuTorch'a dönüştürülür"><defs><marker id="e1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="10" y="70" width="110" height="40" rx="5" class="soft-a" stroke="var(--line)"/><text x="65.0" y="87" text-anchor="middle" class="sm">nn.Module</text><text x="65.0" y="102" text-anchor="middle" class="sm">(Python)</text><rect x="160" y="60" width="150" height="60" rx="5" class="soft-p" stroke="var(--line)"/><text x="235.0" y="77" text-anchor="middle" class="sm">torch.export</text><text x="235.0" y="92" text-anchor="middle" class="sm">ExportedProgram (.pt2)</text><text x="235" y="138" text-anchor="middle" class="sm">ATen grafı + guard’lar</text><text x="235" y="152" text-anchor="middle" class="sm">+ dinamik boyutlar (Dim)</text><path d="M120,90 L156,90" class="ink" marker-end="url(#e1)"/><rect x="370" y="10" width="250" height="46" rx="5" class="soft-b" stroke="var(--line)"/><text x="495.0" y="27" text-anchor="middle" class="sm">AOTInductor paketi</text><text x="495.0" y="42" text-anchor="middle" class="sm">C++ ile çalışır, Python’suz</text><path d="M310,90 L366,33" class="ink" marker-end="url(#e1)"/><rect x="370" y="72" width="250" height="46" rx="5" class="soft-b" stroke="var(--line)"/><text x="495.0" y="89" text-anchor="middle" class="sm">ONNX</text><text x="495.0" y="104" text-anchor="middle" class="sm">ONNX Runtime (sunucu, web)</text><path d="M310,90 L366,95" class="ink" marker-end="url(#e1)"/><rect x="370" y="134" width="250" height="46" rx="5" class="soft-b" stroke="var(--line)"/><text x="495.0" y="151" text-anchor="middle" class="sm">ExecuTorch</text><text x="495.0" y="166" text-anchor="middle" class="sm">mobil ve gömülü</text><path d="M310,90 L366,157" class="ink" marker-end="url(#e1)"/><text x="10" y="196" class="sm">Eski yollar: torch.jit.trace (kontrol akışını dondurur) ve torch.jit.script; yeni kodda torch.export önerilir.</text></svg><figcaption>Şekil 1 — Modeli Python'dan çıkarmanın yolları. Ortak ara adım, yan etkisiz, tam bir ATen grafıdır; hedef çalışma zamanı ondan üretilir.</figcaption></figure>

**El hesapları: boyut ve verim.** 10 milyon parametre fp32'de $\mathbf{40\,000\,000}$ bayt, int8 ağırlıklarla $\mathbf{10\,000\,000}$ bayt (M7.5). Bir model batch 1'de 5 ms, batch 8'de 20 ms sürüyorsa verim batch 1'de $1/0.005=\mathbf{200}$, batch 8'de $8/0.020=\mathbf{400}$ örnek/s'dir. Batch'leme verimi artırır, her isteğin gecikmesini de artırır. Bu ödünleşim modülün sonunda sunucu benzetimiyle inceleniyor.

### İzleme ve tuzağı {#izleme}

Grafı yakalamanın en eski yolu **izlemedir** (tracing). Model örnek bir girdiyle çalıştırılır ve yapılan tensör işlemleri kaydedilir. Ama kayıt yalnız **bu girdide** yapılan işlemleri içerir. Python'un `if`'i, `for`'u ve tensör değerlerinden hesaplanan sayılar grafa girmez, yalnız onların o girdideki sonucu girer.

<figure class="fig wide"><svg viewBox="0 0 640 132" role="img" aria-label="İzleme yalnız örnek girdide alınan dalı kaydeder ve başka girdilerde yanlış sonuç verir; torch.cond iki dalı da grafa koyar"><defs><marker id="e1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><text x="10" y="18" class="ta">İzleme (x₀ = (1, 2), Σx₀ = 3 &gt; 0)</text><rect x="10" y="28" width="190" height="34" rx="5" class="soft-a" stroke="var(--line)"/><text x="105.0" y="49.0" text-anchor="middle" class="sm">x → mul(·, 2) → y</text><text x="10" y="80" class="sm">Koşul somut bir sayıyla hesaplandı, grafa girmedi.</text><text x="10" y="104" class="sm">x = (−1, −1) ile tekrar: y = (−2, −2), Σy = −4</text><text x="10" y="120" class="tb">doğrusu −x = (1, 1), Σ = 2</text><text x="330" y="18" class="tb">torch.cond ile export</text><rect x="330" y="28" width="120" height="34" rx="5" class="soft-p" stroke="var(--line)"/><text x="390.0" y="49.0" text-anchor="middle" class="sm">pred = Σx &gt; 0</text><rect x="480" y="10" width="140" height="30" rx="5" class="soft-a" stroke="var(--line)"/><text x="550.0" y="29.0" text-anchor="middle" class="sm">doğru: x · 2</text><rect x="480" y="50" width="140" height="30" rx="5" class="soft-a" stroke="var(--line)"/><text x="550.0" y="69.0" text-anchor="middle" class="sm">yanlış: −x</text><path d="M450,40 L476,25" class="ink" marker-end="url(#e1)"/><path d="M450,50 L476,65" class="ink" marker-end="url(#e1)"/><text x="330" y="104" class="sm">İki dal da grafta; seçim çalışma anında yapılır.</text><text x="330" y="120" class="sm">Python if ile export ise hata verir (veriye bağlı akış).</text></svg><figcaption>Şekil 2 — Modülün el hesabı: kayıt tabanlı izleyici x₀ = (1, 2) ile 2x dalını kaydeder; x = (−1, −1) için toplam −4 verir, doğrusu 2.</figcaption></figure>

**El hesabı.** $f(x)=2x$ eğer $\sum x>0$, değilse $-x$. İzleyici $x_0=(1,2)$ ile çalışır: $\sum x_0=3>0$, kaydedilen graf "$\times2$"dir. $x=(-1,-1)$ ile tekrar oynatılınca çıktı $(-2,-2)$, toplamı $\mathbf{-4}$; doğrusu $-x=(1,1)$, toplamı $\mathbf2$. `torch.jit.trace` bu durumda yalnız bir uyarı verir ve sessizce yanlış grafı döndürür (ölçtük: $(-2,-2)$).

Tuzak üç biçimde görünür:
- **Veriye bağlı kontrol akışı**: koşul tensör **değerine** bağlıdır (yukarıdaki örnek).
- **Şekle bağlı kontrol akışı**: koşul tensörün **şekline** bağlıdır (`if x.shape[0] > 4`). İzleme örnek şekle özelleşir.
- **Python sabitleri**: tensörden çıkarılan sayılar (`n = x.size(1)`) grafa sabit olarak gömülür; başka bir boyutta graf yanlış olur.

::lab l1

### torch.export: tam graf ve açık kurallar {#export}

`torch.export.export(model, args, dynamic_shapes=...)` (PyTorch 2.x) aynı amaca daha güvenilir bir yoldan ulaşır. Dynamo (M6.5) ile Python bytecode'unu izler, ama torch.compile'dan farklı olarak **graph break'e izin vermez**: grafın tamamı yakalanmalıdır. Sonuç bir `ExportedProgram`'dır:
- **ATen düzeyinde graf**: `aten.linear`, `aten.relu` gibi çekirdek işlemler (ölçtük: bir Linear + ReLU modeli `['aten.linear.default', 'aten.relu.default']` verir). Yan etkisizdir; parametreler ve tamponlar açık girdilerdir.
- **Guard'lar ve kısıtlar**: grafın hangi girdi şekillerinde geçerli olduğu.
- **Dinamik boyutlar**: `Dim("batch", min=1, max=1024)` ile işaretlenen boyutlar sembolik kalır; graf her batch boyunda çalışır (ölçtük: batch 2'de dışa aktarılıp batch 7'de çalıştı). İşaretlenmeyen boyutlar örnek değere özelleşir.

**Veriye bağlı kontrol akışı.** Python `if`'li model dışa aktarılınca hata verir: "Could not guard on data-dependent expression" (`GuardOnDataDependentSymNode`). İzlemenin aksine sessizce yanlış graf üretmez. Çözüm, kontrol akışını grafa açıkça koymaktır:
- `torch.cond(pred, true_fn, false_fn, operands)`: iki dal da alt graf olarak kaydedilir, seçim çalışma anında yapılır. Ölçtük: $x=(-1,-1)$ için doğru sonuç $(1,1)$.
- `torch.where(cond, a, b)`: iki dal da ucuzsa, eleman bazlı seçim (M6.5).
- Döngüler için `torch._higher_order_ops.while_loop`, ya da sabit sayıda yinelemeye açmak.

ExportedProgram `torch.export.save` ile `.pt2` dosyasına yazılır, `torch.export.load` ile okunur; `ep.module()` çağrılabilir bir modül verir (aşama d).

### Graf düzeyinde geçişler {#gecis}

Dışa aktarılan graf, dağıtım öncesi dönüşümlerin yapıldığı yerdir (M6.5'teki FX geçişlerinin dağıtım hâli). Aşama c bunları ONNX benzeri bir mini çalışma zamanında yazıyor.
- **Sabit katlama**: girdilerinin hepsi sabit (ağırlık, sabit tensör) olan düğümler bir kez hesaplanıp sonuç sabit olarak saklanır. Örneğin bir ağırlığın transpozesi ya da iki sabitin çarpımı çalışma anından çıkar.
- **Şekil çıkarımı**: her düğümün çıktı şekli girdi şekillerinden hesaplanır; sembolik boyutlar (`"N"`) korunur. Bellek ayırmayı önceden planlamak ve hataları çalıştırmadan yakalamak için kullanılır. Yayın kuralı: boyutlar eşitse o, biri 1 ise diğeri.
- **Operatör füzyonu ve düzen dönüşümleri**: Conv + BN katlama (M6.5), NCHW'den NHWC'ye geçiş gibi.

### Hedefler: AOTInductor, ONNX, ExecuTorch {#hedef}

- **AOTInductor**: ExportedProgram'ı Inductor ile önceden (ahead-of-time) derler ve bir `.pt2` paketine koyar: derlenmiş çekirdekler (C++/Triton) ve ağırlıklar. Paket C++'tan Python'suz yüklenebilir; Python'dan `torch._inductor.aoti_load_package` ile denenir (aşama e). İlk derleme CPU'da onlarca saniye sürer ve bir C++ derleyicisi gerektirir.
- **ONNX**: çerçeveden bağımsız, düğüm listesi + başlangıç tensörleri (initializers) biçiminde bir graf standardı. `torch.onnx.export(model, args, dynamo=True)` dışa aktarır; ONNX Runtime onu CPU, GPU, mobil ve tarayıcıda (WebAssembly/WebGPU) çalıştırır.
- **ExecuTorch**: mobil ve gömülü cihazlar için PyTorch'un hafif çalışma zamanı; ExportedProgram'dan başlar, donanıma özel arka uçlara (ör. XNNPACK, Core ML, Qualcomm) indirir.
- **Eski yollar**: TorchScript (`torch.jit.trace`, `torch.jit.script`) hâlâ çalışır ama bakım modundadır; yeni kodda `torch.export` önerilir.

### Sunum: batch'leme, gecikme ve verim {#sunum}

Dağıtılmış bir model genellikle birçok istemciden gelen istekleri karşılar. Her isteği tek başına çalıştırmak donanımı boşa harcar: batch başına sabit bir maliyet (çekirdek başlatma, ağırlıkların bellekten okunması; M6.5'teki matris–vektör çarpımı) her istekte yeniden ödenir. **Dinamik batch'leme** gelen istekleri kısa bir süre (ya da batch dolana kadar) bekletip birlikte çalıştırır.

:::türetim Kapasite ve gecikme
Batch süresi $t(B)=t_0+cB$ ise batch başına verim $B/t(B)$'dir; $B\to\infty$'da $1/c$'ye yaklaşır. Modülün el hesabında $t(1)=5$ ms, $t(8)=20$ ms: $c=15/7\approx2.14$ ms, $t_0\approx2.86$ ms. Batch 1 ile kapasite 200 istek/s, batch 8 ile 400. Gelen istek hızı $\lambda$ kapasiteyi aşarsa kuyruk sınırsız büyür: gecikme zamanla artar (kuyruk kuramındaki $\rho=\lambda/\mu\ge1$ durumu). Kapasitenin altında ise gecikme iki parçadır: batch'in dolmasını bekleme (en fazla $w$) ve hesap ($t(B)$).
:::

Aşağıdaki benzetim Poisson isteklerle bu sunucuyu çalıştırıyor. 300 istek/s'de batch'siz sunucu yetişemiyor (kapasite 200), gecikme saniyelere çıkıyor; $B_{\max}=8$ ve $w=5$ ms ile p99 gecikme 39 ms. Dil modeli sunucuları bunun bir adım ötesini yapar: **sürekli batch'leme** (continuous batching) ile her token adımında biten istekleri çıkarıp yenilerini ekler ve KV önbelleğini (M4.7) sayfalara bölerek yönetir (vLLM, PagedAttention).

::lab l2

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| dışa aktarma | `ep = torch.export.export(model, (x,), dynamic_shapes={"x": {0: Dim("batch", min=1, max=1024)}})` |
| graf | `ep.graph_module.print_readable()`; `ep.module()(x)` |
| saklama | `torch.export.save(ep, "m.pt2")`; `torch.export.load("m.pt2")` |
| kontrol akışı | `torch.cond(x.sum() > 0, lambda x: x * 2, lambda x: -x, (x,))` |
| AOTInductor | `path = torch._inductor.aoti_compile_and_package(ep, package_path="m.pt2")`; `torch._inductor.aoti_load_package(path)` |
| ONNX | `torch.onnx.export(model, (x,), "m.onnx", dynamo=True)`; `onnxruntime.InferenceSession("m.onnx")` |
| eski izleme | `torch.jit.trace(model, x)` (kontrol akışını dondurur) |

### Özet {#ozet}

- Dışa aktarma modeli Python'dan bağımsız, sabit bir grafa çevirir. El hesapları: 10 milyon parametre 40 MB (fp32), 10 MB (int8); verim batch 1'de 200, batch 8'de 400 örnek/s.
- İzleme yalnız örnek girdide alınan yolu kaydeder: veriye/şekle bağlı akış ve Python sabitleri dondurulur (el hesabı: −4 yerine 2 olmalı). `jit.trace` sessizce yanlış graf verir.
- `torch.export`: tam ATen grafı, guard'lar, `Dim` ile dinamik boyutlar; veriye bağlı `if`'te hata verir; çözüm `torch.cond`/`torch.where`.
- Graf geçişleri: sabit katlama, şekil çıkarımı (sembolik boyutlar, yayın), füzyon.
- Hedefler: AOTInductor paketi (Python'suz), ONNX (Runtime, web), ExecuTorch (mobil).
- Sunumda dinamik batch'leme verimi artırır, beklemeyi getirir; λ kapasiteyi aşınca kuyruk patlar. LLM'lerde sürekli batch'leme.
