### Neden düşük hassasiyet? {#neden}

Modern hızlandırıcıların tensör çekirdekleri fp32'den çok daha hızlı 16 bitlik (ve 8 bitlik) matris çarpımı yapar: A100'de bf16/fp16 ≈ 312 TFLOP/s, fp32 (tensör çekirdeksiz) ≈ 19.5 TFLOP/s. Ayrıca 16 bitlik bir tensör yarı bellek tutar ve bellek bağımlı işlemlerde (M6.5) yarı sürede taşınır. Ama düşük hassasiyetle eğitmek üç yeni hata kaynağı getirir: **taşma** (sayı çok büyük), **alt taşma** (sayı çok küçük) ve **kaybolan güncellemeler** (küçük bir değişikliğin büyük bir sayıya eklenince yuvarlanıp yok olması). Karışık hassasiyetli eğitim (Micikevicius ve ark. 2018), her işlemi güvenli olduğu en düşük hassasiyette yapıp bu üç sorunu ayrı ayrı çözer.

### Kayan nokta biçimleri {#bicim}

Bir IEEE tarzı sayı $x=(-1)^s\cdot(1.f)_2\cdot2^{E-\text{bias}}$ biçimindedir: 1 işaret biti, $e$ üs biti, $m$ mantis (kesir) biti, $\text{bias}=2^{e-1}-1$.

<figure class="fig wide"><svg viewBox="0 0 640 228" role="img" aria-label="Kayan nokta biçimlerinin bit düzeni: fp32 8 üs 23 mantis, tf32 8 ve 10, bf16 8 ve 7, fp16 5 ve 10, fp8 e5m2 ve e4m3; en büyük değerler ve eps"><text x="420" y="16" class="sm">en büyük</text><text x="520" y="16" class="sm">eps (2⁻ᵐ)</text><text x="82" y="39" text-anchor="end" class="sm">fp32</text><rect x="90" y="24" width="7.6" height="20" class="soft-b" stroke="var(--line)"/><rect x="98.6" y="24" width="67.8" height="20" class="soft-p" stroke="var(--line)"/><text x="133.0" y="38" text-anchor="middle" class="sm">8</text><rect x="167.39999999999998" y="24" width="196.79999999999998" height="20" class="soft-a" stroke="var(--line)"/><text x="266.29999999999995" y="38" text-anchor="middle" class="sm">23</text><text x="420" y="39" class="sm">3.4·10³⁸</text><text x="520" y="39" class="sm">1.2·10⁻⁷</text><text x="82" y="69" text-anchor="end" class="sm">tf32</text><rect x="90" y="54" width="7.6" height="20" class="soft-b" stroke="var(--line)"/><rect x="98.6" y="54" width="67.8" height="20" class="soft-p" stroke="var(--line)"/><text x="133.0" y="68" text-anchor="middle" class="sm">8</text><rect x="167.39999999999998" y="54" width="85.0" height="20" class="soft-a" stroke="var(--line)"/><text x="210.39999999999998" y="68" text-anchor="middle" class="sm">10</text><text x="420" y="69" class="sm">3.4·10³⁸</text><text x="520" y="69" class="sm">9.8·10⁻⁴</text><text x="82" y="99" text-anchor="end" class="sm">bf16</text><rect x="90" y="84" width="7.6" height="20" class="soft-b" stroke="var(--line)"/><rect x="98.6" y="84" width="67.8" height="20" class="soft-p" stroke="var(--line)"/><text x="133.0" y="98" text-anchor="middle" class="sm">8</text><rect x="167.39999999999998" y="84" width="59.199999999999996" height="20" class="soft-a" stroke="var(--line)"/><text x="197.49999999999997" y="98" text-anchor="middle" class="sm">7</text><text x="420" y="99" class="sm">3.4·10³⁸</text><text x="520" y="99" class="sm">7.8·10⁻³</text><text x="82" y="129" text-anchor="end" class="sm">fp16</text><rect x="90" y="114" width="7.6" height="20" class="soft-b" stroke="var(--line)"/><rect x="98.6" y="114" width="42.0" height="20" class="soft-p" stroke="var(--line)"/><text x="120.1" y="128" text-anchor="middle" class="sm">5</text><rect x="141.6" y="114" width="85.0" height="20" class="soft-a" stroke="var(--line)"/><text x="184.6" y="128" text-anchor="middle" class="sm">10</text><text x="420" y="129" class="sm">65 504</text><text x="520" y="129" class="sm">9.8·10⁻⁴</text><text x="82" y="159" text-anchor="end" class="sm">fp8 e5m2</text><rect x="90" y="144" width="7.6" height="20" class="soft-b" stroke="var(--line)"/><rect x="98.6" y="144" width="42.0" height="20" class="soft-p" stroke="var(--line)"/><text x="120.1" y="158" text-anchor="middle" class="sm">5</text><rect x="141.6" y="144" width="16.2" height="20" class="soft-a" stroke="var(--line)"/><text x="150.2" y="158" text-anchor="middle" class="sm">2</text><text x="420" y="159" class="sm">57 344</text><text x="520" y="159" class="sm">0.25</text><text x="82" y="189" text-anchor="end" class="sm">fp8 e4m3</text><rect x="90" y="174" width="7.6" height="20" class="soft-b" stroke="var(--line)"/><rect x="98.6" y="174" width="33.4" height="20" class="soft-p" stroke="var(--line)"/><text x="115.8" y="188" text-anchor="middle" class="sm">4</text><rect x="133.0" y="174" width="24.799999999999997" height="20" class="soft-a" stroke="var(--line)"/><text x="145.9" y="188" text-anchor="middle" class="sm">3</text><text x="420" y="189" class="sm">448</text><text x="520" y="189" class="sm">0.125</text><rect x="90" y="208" width="12" height="12" class="soft-b"/><text x="106" y="218" class="sm">işaret</text><rect x="160" y="208" width="12" height="12" class="soft-p"/><text x="176" y="218" class="sm">üs: aralık</text><rect x="260" y="208" width="12" height="12" class="soft-a"/><text x="276" y="218" class="sm">mantis: hassasiyet</text></svg><figcaption>Şekil 1 — Üs bitleri aralığı, mantis bitleri hassasiyeti belirler. bf16, fp32'nin aralığını korur ama hassasiyetin çoğunu atar; fp16 tersini yapar. Değerler laboratuvardaki yuvarlama koduyla hesaplandı.</figcaption></figure>

:::türetim Aralık ve hassasiyet bitlerden nasıl çıkar?
- **Hassasiyet**: $[1,2)$ aralığında ardışık iki sayının farkı, mantisin son bitinin değeri, $2^{-m}$'dir (makine epsilonu). fp16'da $2^{-10}=\mathbf{0.000977}$, bf16'da $2^{-7}=\mathbf{0.0078}$. Göreli yuvarlama hatası en fazla $2^{-m-1}$'dir. Mutlak aralık büyüklükle orantılıdır: $[2^E,2^{E+1})$'de adım $2^{E-m}$.
- **Aralık**: en büyük üs (tümü 1 olan kod özel değerler için ayrılmışsa) $\text{bias}$'tır, en büyük sonlu değer $(2-2^{-m})\cdot2^{\text{bias}}$. fp16'da $(2-2^{-10})\cdot2^{15}=\mathbf{65\,504}$; bf16'da fp32 ile aynı ≈ $3.4\cdot10^{38}$.
- **Alt normaller**: en küçük normal sayı $2^{1-\text{bias}}$ (fp16'da $2^{-14}\approx6.1\cdot10^{-5}$). Altında baştaki 1 düşer ve hassasiyet azalarak $2^{1-\text{bias}-m}$'ye (fp16'da $2^{-24}\approx5.96\cdot10^{-8}$) kadar inilir; bunun yarısından küçükler 0 olur.
:::

**El hesapları.** $300^2=90\,000>65\,504$: fp16'da **taşar** (sonuç $\infty$). Logitler, dikkat skorları ya da kareli toplamlar (varyans) bu yüzden fp16'da tehlikelidir. bf16'da taşmaz ama yalnız ~2–3 anlamlı ondalık basamak taşır.

**bf16 mı, fp16 mı?** bf16 (Google Brain'in "beyin" kayan noktası; Kalamkar ve ark. 2019) fp32'nin üs alanını korur: taşma ve alt taşma pratikte yoktur, kayıp ölçekleme gerekmez. Bedeli 3 bit daha az mantistir. fp16 daha hassastır ama dar aralığı kayıp ölçekleme gerektirir. A100 ve sonrası her ikisini aynı hızda yaptığı için bugün eğitimde çoğunlukla bf16 kullanılır. tf32, fp32 tensörlerini matris çarpımında 10 bitlik mantisle hesaplayan bir tensör çekirdeği modudur (`torch.backends.cuda.matmul.allow_tf32`). fp8 (e4m3: ileri geçiş, e5m2: gradyanlar; Micikevicius ve ark. 2022) en yeni donanımlarda tensör başına ölçek çarpanlarıyla kullanılır.

::lab l1

### Kaybolan güncellemeler ve fp32 ana kopya {#master}

$w\leftarrow w-\eta g$ güncellemesi, $|\eta g|$ ağırlığın komşu sayıya uzaklığının yarısından küçükse **hiçbir şey yapmaz**: sonuç yine $w$'ye yuvarlanır.

**El hesabı.** fp16'da $w=1$: komşu sayı $1+2^{-10}$, yarım aralık $\approx4.9\cdot10^{-4}$. $10^{-4}$'lük bir güncelleme kaybolur: $1+10^{-4}=\mathbf{1}$. Öğrenme oranları $10^{-4}$, gradyanlar $10^{-2}$ mertebesindeyken güncellemeler $10^{-6}$ civarındadır. Ağırlıkları fp16'da tutan bir eğitim bu yüzden takılır (aşama c).

Çözüm, ağırlıkların bir **fp32 ana kopyasını** tutmaktır. İleri ve geri geçiş 16 bitlik bir kopya kullanır; güncelleme fp32 kopyaya yapılır ve küçük katkılar orada birikir. Bellek bedeli parametre başına 4 bayt daha (M6.4'teki 16 baytın parçası).

### Alt taşma ve kayıp ölçekleme {#scaling}

fp16'da $2^{-25}$'ten küçük gradyanlar sıfır olur. Micikevicius ve ark., bazı ağlarda aktivasyon gradyanlarının önemli bir kısmının bu bölgede olduğunu gösterdi. Bu gradyanlar yok olunca bilgi kaybolur.

:::türetim Ölçeklemenin neden işe yaradığı
Geri geçiş doğrusaldır: kayıp $S$ ile çarpılırsa her ara gradyan da $S$ ile çarpılır ($\partial(SL)/\partial v=S\,\partial L/\partial v$). Gradyan dağılımı $\log_2S$ kadar yukarı kayar ve alt taşma bölgesinden çıkar. Güncellemeden önce gradyanlar fp32'de $S$'ye bölünür; sonuç ölçeksiz gradyanla aynıdır, yalnız yolda kaybolmamıştır. $S$ 2'nin kuvveti seçilir: çarpma ve bölme yalnız üssü değiştirir, mantis bitleri olduğu gibi kalır.

El hesabı: $10^{-8}$'lik bir gradyan fp16'da 0 olur ($<2^{-25}$). En küçük pozitif fp16 değeri $2^{-24}\approx5.96\cdot10^{-8}$'in üstüne çıkaran en küçük 2'nin kuvveti ölçek $\mathbf8$'dir ($8\cdot10^{-8}$).
:::

**Dinamik ölçek.** $S$ çok büyükse bu kez en büyük gradyanlar taşar ($\infty$ ya da NaN). `GradScaler` ölçeği kendisi ayarlar: her adımda ölçeklenmiş gradyanlarda inf/NaN arar. Bulursa o adımı **atlar** (optimizasyon adımı yapılmaz) ve $S$'yi yarıya indirir. Belirli sayıda (varsayılan 2000) ardışık iyi adımdan sonra $S$'yi iki katına çıkarır. Varsayılan başlangıç $2^{16}$'dır. Böylece $S$ taşmanın hemen altında dolaşır.

<figure class="fig wide"><svg viewBox="0 0 640 176" role="img" aria-label="Karışık hassasiyetli eğitim adımı: fp32 ana ağırlıklar, autocast'li ileri geçiş, kaybın S ile ölçeklenmesi, fp16 gradyanlar, ölçeğin geri alınması ve inf denetimi, fp32 güncelleme"><defs><marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="10" y="22" width="120" height="32" rx="4" class="soft-a" stroke="var(--line)"/><text x="70.0" y="42" text-anchor="middle" class="sm">fp32 ana ağırlıklar</text><rect x="156" y="22" width="300" height="32" rx="4" class="soft-p" stroke="var(--line)"/><text x="306.0" y="42" text-anchor="middle" class="sm">autocast ileri: matmul fp16/bf16, softmax ve kayıp fp32</text><rect x="482" y="22" width="120" height="32" rx="4" class="soft-b" stroke="var(--line)"/><text x="542.0" y="42" text-anchor="middle" class="sm">kayıp × S</text><rect x="10" y="92" width="190" height="32" rx="4" class="soft-p" stroke="var(--line)"/><text x="105.0" y="112" text-anchor="middle" class="sm">geri geçiş (fp16 gradyanlar)</text><rect x="226" y="92" width="160" height="32" rx="4" class="soft-b" stroke="var(--line)"/><text x="306.0" y="112" text-anchor="middle" class="sm">÷ S, inf/NaN denetimi</text><rect x="412" y="92" width="220" height="32" rx="4" class="soft-a" stroke="var(--line)"/><text x="522.0" y="112" text-anchor="middle" class="sm">fp32 güncelleme ya da adımı atla</text><path d="M132,38 L153,38" class="ink" marker-end="url(#a1)"/><path d="M458,38 L479,38" class="ink" marker-end="url(#a1)"/><path d="M542.0,54 L542.0,68 L105.0,68 L105.0,89" class="ink" marker-end="url(#a1)"/><path d="M202,108 L223,108" class="ink" marker-end="url(#a1)"/><path d="M388,108 L409,108" class="ink" marker-end="url(#a1)"/><text x="10" y="150" class="sm">S dinamik: taşmada adım atlanır, S yarıya iner; 2000 ardışık iyi adımda iki katına çıkar (GradScaler).</text><text x="10" y="166" class="sm">bf16’da aralık fp32 ile aynı olduğu için S gerekmez; autocast ve (çoğu zaman) fp32 ana ağırlıklar yeter.</text></svg><figcaption>Şekil 2 — AMP'nin üç parçası: autocast (işlem başına tip seçimi), fp32 ana ağırlıklar ve kayıp ölçekleme (Micikevicius ve ark. 2018).</figcaption></figure>

::lab l2

### autocast: işlem başına tip {#autocast}

Her işlem düşük hassasiyete uygun değildir. `torch.autocast(device_type, dtype)` bağlamı, bir listeye göre her işlemin girdilerini uygun tipe çevirir:
- **16 bitte**: matris çarpımları, `linear`, konvolüsyonlar, `bmm`. Tensör çekirdeklerinden yararlanır; birikim donanımda fp32'de yapılır.
- **fp32'de**: `softmax`, `log_softmax`, `layer_norm`, kayıp fonksiyonları (`cross_entropy`), `exp`, `log`, büyük indirgemeler. Bunlar ya büyük değerler (exp), ya çok sayıda küçük katkının toplamı (indirgemeler), ya da log gibi hassas işlemlerdir.
- **Girdiye uyan**: eleman bazlı işlemler, girdilerinden en geniş tipte çalışır.

Parametreler fp32 kalır (autocast yalnız işlem anında bir kopya kullanır); gradyanlar parametrelerle aynı tipte (fp32) birikir. Geri geçiş, ileri geçişteki tiplerle otomatik çalışır; `backward` autocast bağlamının **dışında** çağrılır. Aşama d, bir modelin çıktısının, kaybının, parametresinin ve gradyanının tiplerini autocast içinde ölçüyor.

### Saf bf16 eğitim ve stokastik yuvarlama {#sr}

Bellek çok kısıtlıysa fp32 ana kopya da atılıp ağırlıklar doğrudan bf16 tutulabilir. O zaman kaybolan güncelleme sorunu geri gelir (bf16'da daha da kötü: 3 bit daha az mantis). **Stokastik yuvarlama** (Gupta ve ark. 2015) bunu çözer: $x$, iki komşu temsil edilebilir sayı $a\le x<b$ arasında, uzaklıklarla orantılı olasılıkla yuvarlanır.

:::türetim Stokastik yuvarlama yansızdır
$p=\frac{x-a}{b-a}$ olasılıkla $b$'ye, $1-p$ ile $a$'ya yuvarla:
$$\mathbb E[\mathrm{SR}(x)]=a(1-p)+bp=a+(b-a)\frac{x-a}{b-a}=x .$$
En yakına yuvarlama ise deterministiktir ve küçük bir güncelleme için hep aynı yöne (geri, $w$'ye) yanlıdır. SR'de güncelleme küçük bile olsa **beklentide** uygulanır: binlerce adım boyunca birikir. Bedeli, yuvarlamanın getirdiği gürültüdür (varyans $\le(b-a)^2/4$).

**Bit hilesi** (aşama e). fp32 bir sayının üst 16 biti bf16'dır. Alt 16 bite $[0,2^{16})$'dan düzgün bir rastgele tamsayı ekleyip alt 16 biti sıfırlamak tam olarak SR'dir. Elde (carry) üst bitlere geçme olasılığı, alt 16 bitin değeri / $2^{16}$, yani $p$'dir. İşaret biti ayrı olduğu için büyüklük üzerinde çalışır ve negatif sayılarda da doğrudur.
:::

Aşağıdaki deneyde bf16 ağırlıkla basit bir ikinci dereceden kaybı küçük öğrenme oranıyla en küçükle. En yakına yuvarlama ilk adımda takılır; stokastik yuvarlama fp32 yörüngesini izler.

::lab l3

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| bf16 autocast (GPU) | `with torch.autocast("cuda", dtype=torch.bfloat16): loss = F.cross_entropy(model(x), y)` |
| fp16 + ölçekleme | `scaler = torch.amp.GradScaler("cuda")`; `scaler.scale(loss).backward(); scaler.step(opt); scaler.update()` |
| gradyan kırpma ile | `scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` sonra `scaler.step(opt)` |
| CPU'da | `torch.autocast("cpu", dtype=torch.bfloat16)` |
| tf32 | `torch.backends.cuda.matmul.allow_tf32 = True` ya da `torch.set_float32_matmul_precision("high")` |
| biçim bilgisi | `torch.finfo(torch.float16).max`, `.eps`, `.tiny` |
| SR (bit hilesi) | `((x.view(torch.int32) + torch.randint(0, 1 << 16, x.shape, dtype=torch.int32)) & -65536).view(torch.float32)` |

### Özet {#ozet}

- Biçim: $e$ üs biti aralığı, $m$ mantis biti hassasiyeti belirler; eps $=2^{-m}$ (fp16 0.000977, bf16 0.0078), fp16'nın en büyüğü 65 504 ($300^2$ taşar).
- Kaybolan güncelleme: $|\Delta|<\text{ulp}(w)/2$ ise $w+\Delta=w$ (fp16'da $1+10^{-4}=1$); çözüm fp32 ana kopya.
- Kayıp ölçekleme: geri geçiş doğrusal, $S$ ile çarp, fp32'de böl; $10^{-8}$ için en küçük ölçek 8. Dinamik ölçek: inf'te adımı atla ve yarıya indir, 2000 iyi adımda ikiye katla.
- autocast: matmul/konvolüsyon 16 bit, softmax/norm/kayıp/exp fp32; parametreler ve gradyanlar fp32.
- bf16: fp32 aralığı, ölçekleme gereksiz; eğitimde varsayılan tercih. fp8: e4m3/e5m2 ve tensör başına ölçekler.
- Saf bf16 eğitimde stokastik yuvarlama yansızdır ($\mathbb E[\mathrm{SR}(x)]=x$); bit hilesi: alt 16 bite rastgele ekle, kes.
