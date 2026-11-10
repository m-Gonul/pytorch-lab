### Bitirme projesi: iki iz, bir mimari {#proje}

Bu modül kursun görüntü tarafını tek bir araştırma prototipinde birleştiriyor. Konvolüsyon ve U-Net (M4.10), normalizasyon (M3.6), dikkat (M4.6), difüzyon ve yönlendirme (M5.3, M5.8), eğitim döngüsü ve profil (M8.1) ile değerlendirme (M5.7) burada tek bir çalışan sisteme bağlanıyor. İki iz var, ikisi de aynı U-Net ailesini kullanıyor:
- **Üretim**: rakam sınıfıyla koşullu, sınıflandırıcısız yönlendirmeli bir difüzyon modeli.
- **Analiz**: ön plan piksellerinin yalnız ~%5 olduğu küçük "lezyon" görüntülerinde segmentasyon.

Bir araştırma prototipinden beklenen, yalnız çalışan bir model değil, **dürüst bir değerlendirmedir**: taban çizgileri, güven aralıkları ve neyin işe yaramadığının raporu.

### Difüzyon U-Net'i {#unet}

Ho ve ark.'nın (2020) DDPM'de kullandığı, PixelCNN++ tabanlı U-Net'in küçültülmüş hâli:

<figure class="fig wide"><svg viewBox="0 0 640 258" role="img" aria-label="Küçük difüzyon U-Net'i: 16×16'dan 8×8'e ve 4×4'e inen kodlayıcı, en dipte öz-dikkat, yukarı çıkarken atlama bağlantılarıyla birleştirme; zaman ve sınıf embedding'i her artık bloka eklenir"><defs><marker id="u1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker><marker id="u2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><rect x="10" y="20" width="110" height="36" rx="4" class="soft-a" stroke="var(--line)"/><text x="65.0" y="42.0" text-anchor="middle" class="sm">x<tspan baseline-shift="-3" font-size="75%">t</tspan> (1 × 16 × 16)</text><rect x="10" y="70" width="110" height="36" rx="4" class="soft-p" stroke="var(--line)"/><text x="65.0" y="85" text-anchor="middle" class="sm">ResBlock c = 16</text><text x="65.0" y="99" text-anchor="middle" class="sm">16 × 16</text><rect x="150" y="120" width="110" height="36" rx="4" class="soft-p" stroke="var(--line)"/><text x="205.0" y="135" text-anchor="middle" class="sm">ResBlock 2c = 32</text><text x="205.0" y="149" text-anchor="middle" class="sm">8 × 8</text><rect x="290" y="170" width="130" height="36" rx="4" class="soft-p" stroke="var(--line)"/><text x="355.0" y="185" text-anchor="middle" class="sm">ResBlock + öz-dikkat</text><text x="355.0" y="199" text-anchor="middle" class="sm">4 × 4: 16 konum</text><rect x="450" y="120" width="130" height="36" rx="4" class="soft-p" stroke="var(--line)"/><text x="515.0" y="135" text-anchor="middle" class="sm">birleştir 32 + 32</text><text x="515.0" y="149" text-anchor="middle" class="sm">8 × 8</text><rect x="520" y="70" width="110" height="36" rx="4" class="soft-p" stroke="var(--line)"/><text x="575.0" y="85" text-anchor="middle" class="sm">birleştir 32 + 16</text><text x="575.0" y="99" text-anchor="middle" class="sm">= 48 kanal</text><rect x="520" y="20" width="110" height="36" rx="4" class="soft-a" stroke="var(--line)"/><text x="575.0" y="42.0" text-anchor="middle" class="sm">ε<tspan dx="-0.39em" dy="-0.2em">ˆ</tspan><tspan dx="0.04em" dy="0.2em"> (1 × 16 × 16)</tspan></text><path d="M65,56 L65,68" class="ink" marker-end="url(#u1)"/><path d="M100,106 L160,118" class="ink" marker-end="url(#u1)"/><path d="M240,156 L300,168" class="ink" marker-end="url(#u1)"/><path d="M420,182 L470,158" class="ink" marker-end="url(#u1)"/><path d="M560,120 L570,108" class="ink" marker-end="url(#u1)"/><path d="M575,70 L575,58" class="ink" marker-end="url(#u1)"/><path d="M120,88 C300,80 400,80 518,88" class="cb dash" marker-end="url(#u2)"/><text x="320" y="76" text-anchor="middle" class="tb">atlama (16 kanal)</text><path d="M260,138 L448,138" class="cb dash" marker-end="url(#u2)"/><text x="355" y="132" text-anchor="middle" class="tb">atlama (32)</text><text x="10" y="232" class="sm">t → sinüzoidal → MLP; y → embedding (C + 1, sonuncusu boş token); toplam her ResBlock’a eklenir.</text><text x="10" y="248" class="sm">ResBlock: GN → SiLU → konv → + emb → GN → SiLU → konv (sıfır başl.) → + girdi. Dikkat skoru: 256 eleman.</text></svg><figcaption>Şekil 1 — Bitirme projesinin U-Net'i (c = 16). El hesapları: 4 × 4'te öz-dikkat 16 konum, 256 skor; en üst kod çözücü bloğunun girdisi 2c + c = 48 kanal.</figcaption></figure>

- **Artık bloklar**: GroupNorm → SiLU → 3×3 konvolüsyon → (zaman + sınıf embedding'inin doğrusal izdüşümü, kanal başına eklenir) → GroupNorm → SiLU → 3×3 konvolüsyon → girdiye eklenir. GroupNorm, BatchNorm'un aksine batch istatistiklerine bağlı değildir (M3.6); difüzyonda her örnek farklı bir gürültü düzeyinde olduğu için batch istatistikleri anlamsızdır.
- **Sıfır başlatılmış çıkış**: her bloğun ikinci konvolüsyonu (ve ağın son katmanı) sıfırla başlatılır. Başlangıçta her blok özdeşliktir ve ağ derin olsa da eğitim kararlı başlar (M4.2'deki artık bağlantı fikri).
- **Zaman embedding'i**: $t$, transformer'ın konum kodlamasıyla aynı sinüzoidal fonksiyonla (M4.14) vektöre çevrilir, bir MLP'den geçer. El hesabı: $d=8$, ilk frekans $\omega_1=10000^{-1/4}=0.1$, $t=50$ için $\sin(\omega_1t)=\sin5=\mathbf{-0.959}$.
- **Öz-dikkat** yalnız en düşük çözünürlükte: 16 × 16 girdi iki kez yarıya inince 4 × 4 = 16 konum, skor matrisi $16^2=\mathbf{256}$ eleman. Yüksek çözünürlükte dikkatin maliyeti konum sayısının karesiyle büyüdüğü için oraya konmaz.
- **Atlama bağlantıları**: kodlayıcının her çözünürlükteki çıktısı kod çözücüde aynı çözünürlüğe birleştirilir (M4.10). El hesabı: $c=16$, en üst kod çözücü bloğunun girdisi yukarı örneklenmiş $2c$ ile atlamadan gelen $c$: $\mathbf{48}$ kanal.

Aşama c bu blokları NumPy'da (GroupNorm, 3×3 konvolüsyon, artık blok, 2-B dikkat) yazıyor; aşama d PyTorch'ta U-Net'i kurup eğitiyor.

### Eğitim: CFG, EMA {#egitim}

Kayıp M5.3'teki $\varepsilon$-tahmini MSE'sidir. İki ek parça vardır:
- **Etiket bırakma**: etiketlerin %10'u boş token'a çevrilir; tek ağ hem koşullu hem koşulsuz tahmin yapar (M5.8).
- **EMA ağırlıkları**: her adımda $\bar\theta\leftarrow\beta\bar\theta+(1-\beta)\theta$. Örnekleme EMA ağırlıklarıyla yapılır. SGD'nin son adımlardaki gürültüsü ağırlıklarda kalır ve difüzyon örneklerinde gözle görülür kalite kaybına yol açar; EMA bu gürültünün ortalamasını alır.

:::türetim EMA'nın penceresi
Açarak yazınca $\bar\theta_t=(1-\beta)\sum_{k\ge0}\beta^k\theta_{t-k}$ (başlangıç terimi hariç): $k$ adım önceki ağırlığın ağırlığı $(1-\beta)\beta^k$, geometrik azalır. Ağırlıkların ortalama yaşı $\sum_kk(1-\beta)\beta^k=\beta/(1-\beta)\approx1/(1-\beta)$: **etkin pencere**. El hesapları: $\beta=0.995$'in penceresi $\mathbf{200}$ adım; $\beta=0.999$ ile 1000 adım önceki ağırlığın payı $0.999^{1000}=\mathbf{0.368}\approx1/e$. EMA 0'dan başlatılırsa ilk adımlar 0'a doğru yanlıdır; Adam'daki gibi $\bar\theta_t/(1-\beta^t)$ ile düzeltilir ya da EMA ilk ağırlıklarla başlatılır.
:::

Pencere bir ödünleşimdir: uzun pencere gürültüyü daha çok bastırır ama değişikliklere geç yetişir. Kısa eğitimlerde çok büyük $\beta$ seçilirse EMA henüz eğitilmemiş erken ağırlıkları taşır. Aşağıdaki deney bunu bir parametrenin yörüngesinde gösteriyor.

::lab l1

### Örnekleme maliyeti {#ornekleme}

Çıkarımın maliyeti ağ değerlendirmesi sayısıdır (NFE).

<figure class="fig wide"><svg viewBox="0 0 640 190" role="img" aria-label="Çıkarım maliyeti: DDPM 200 adım artı CFG 400 ağ değerlendirmesi, DDIM 25 adım artı CFG 50, toplu CFG ile 25 çağrı"><text x="226" y="33" text-anchor="end" class="sm">DDPM 200 adım + CFG, iki çağrı</text><rect x="234" y="16" width="320" height="24" class="soft-b" stroke="var(--line)"/><text x="560" y="33" class="ink">400</text><text x="226" y="67" text-anchor="end" class="sm">DDIM 25 adım + CFG, iki çağrı</text><rect x="234" y="50" width="40" height="24" class="soft-p" stroke="var(--line)"/><text x="280" y="67" class="ink">50</text><text x="226" y="101" text-anchor="end" class="sm">DDIM 25 adım + CFG, tek batch</text><rect x="234" y="84" width="20" height="24" class="soft-a" stroke="var(--line)"/><text x="260" y="101" class="ink">25</text><text x="226" y="135" text-anchor="end" class="sm">DDIM 25 adım, CFG yok</text><rect x="234" y="118" width="20" height="24" class="soft-a" stroke="var(--line)"/><text x="260" y="135" class="ink">25</text><text x="10" y="164" class="sm">Toplu CFG’de çağrılar yarıya iner, batch iki katına çıkar: paralellik yetiyorsa süre ≈ yarıya iner.</text><text x="10" y="180" class="sm">Bütün örnekler EMA ağırlıklarıyla üretilir; EMA bozunması 0.999 ile 1000 adım önceki ağırlığın payı 0.999¹⁰⁰⁰ ≈ 0.368.</text></svg><figcaption>Şekil 2 — Ağ değerlendirmesi sayısı (NFE). El hesabı: DDIM 25 adım + CFG (ayrı çağrılar) = 50.</figcaption></figure>

- **DDPM** her adımda gürültü ekleyerek T = 200 adımda örnekler (aşama e).
- **DDIM** deterministiktir ve eşit aralıklı daha az adımla (ör. 25) çalışır (M5.3). Aşama b, T − 1'den 0'a eşit aralıklı adımları üretiyor.
- **CFG** her adımda koşullu ve koşulsuz iki tahmin ister. El hesabı: DDIM 25 adım + CFG, ayrı çağrılarla $\mathbf{50}$ NFE. İki girdiyi tek bir batch'te birleştirmek (koşullu ve boş etiketler) çağrı sayısını yarıya indirir, sonucu değiştirmez.
- **Eşikleme**: her adımda tahmin edilen $\hat x_0$, veri aralığına ($[-1,1]$) kırpılır. Büyük yönlendirme ölçeklerinde aşırı doymayı sınırlar (M5.8).

### Değerlendirme disiplini {#degerlendirme}

- **FID'i taban çizgisiyle raporla.** Sınıflandırıcı özniteliklerinde gerçek–gerçek FID (aynı boyutta iki gerçek örneklem) sıfır değildir; sonlu örneklem yanlılığıdır (M5.7). Modelin FID'i bu tabanla birlikte verilmelidir.
- **Sınıf doğruluğunu ayrı ölç.** Koşullu bir model güzel ama yanlış sınıftan örnekler üretebilir; bir sınıflandırıcıyla üretilen örneklerin sınıf doğruluğu ölçülür.
- **Yönlendirme ölçeğini tara.** $s$ kaliteyi (sınıf doğruluğu, kesinlik) ve çeşitliliği (FID, geri çağırma) farklı yönlere iter (M5.8); tek bir $s$ değeri sonuç değildir.

### Segmentasyon izi: dengesiz sınıflar {#segmentasyon}

Ön plan piksellerinin ~%5 olduğu görüntülerde, her pikseli "arka plan" diyen bir model %95 piksel doğruluğu alır ama hiçbir lezyonu bulmaz. Ölçüt ve kayıp bu yüzden önemlidir (M4.10).
- **Dice** (F1'in piksel karşılığı): $\frac{2|P\cap G|}{|P|+|G|}$. El hesabı: tahmin 30 piksel, gerçek 40, kesişim 25: $\frac{50}{70}=\mathbf{0.714}$.
- **BCE**: her pikseli bağımsız sınıflar; az olan ön plan piksellerinin kayba katkısı küçüktür ve model temkinli (düşük duyarlılık) olur. **Ağırlıklı BCE** ön plana daha çok ağırlık verir.
- **Dice kaybı** (yumuşak Dice, Milletari ve ark. 2016) doğrudan örtüşmeyi en büyükler ve sınıf oranından etkilenmez; ama küçük nesnelerde gradyanı gürültülüdür. **BCE + Dice** (nnU-Net'in varsayılanı; Isensee ve ark. 2021) ikisinin dengesidir.

Kayıp seçimi hassasiyet (yanlış alarm azlığı) ile duyarlılık (kaçırılan lezyon azlığı) arasında bir seçimdir; tıbbi uygulamada hangisinin daha pahalı olduğu bu seçimi belirler. Ek x1 dört kaybı aynı veride karşılaştırıyor.

### İki modeli dürüstçe karşılaştırmak {#bootstrap}

İki model (ör. BCE ile Dice kaybı) aynı test kümesinde 0.71 ve 0.73 ortalama Dice aldıysa, fark gerçek mi? Test örneklerinin zorluğu iki modelde ortaktır: zor bir görüntüde ikisi de kötü, kolay birinde ikisi de iyidir.

:::türetim Eşleştirilmiş bootstrap
Örnek başı skorlar $a_i$, $b_i$ ve farklar $d_i=b_i-a_i$ olsun. Ortalama farkın varyansı
$$\mathrm{Var}(\bar d)=\frac{\sigma_a^2+\sigma_b^2-2\rho\sigma_a\sigma_b}n ,$$
iki skorun ilişkisi $\rho$ büyüdükçe küçülür. Eşleştirilmemiş karşılaştırma (iki ortalamanın farkı, ayrı ayrı yeniden örneklenmiş) bu ortak gürültüyü yok sayar: $\mathrm{Var}=(\sigma_a^2+\sigma_b^2)/n$. Eşleştirilmiş bootstrap test örneklerini **aynı indekslerle** yeniden örnekler ve $\bar d$'nin dağılımından yüzdelik bir güven aralığı verir. Eşit varyansta genişlik oranı $\approx\sqrt{1-\rho}$'dur.
:::

Aşağıdaki deneyde $\rho=0.9$ ve $n=100$ ile 0.02'lik bir fark eşleştirilmiş aralıkta açıkça 0'dan ayrılıyor ([0.016, 0.045]), eşleştirilmemiş aralık ise 0'ı içeriyor. Eğitimin kendi rastgeleliği (tohum) de bir varyans kaynağıdır; ciddi bir karşılaştırma birkaç tohumla tekrar edilir. Ek x3 bunu gerçek segmentasyon skorlarıyla yapıyor.

::lab l2

### Araştırma pratiği: difüzyonla tamamlama {#repaint}

Eğitilmiş bir koşulsuz difüzyon modeliyle bir görüntünün eksik bölgesini doldurmak (inpainting) cazip görünür: her adımda bilinen pikselleri, ileri süreçle aynı gürültü düzeyine getirilmiş gerçek değerlerle değiştir, bilinmeyenleri modelin ürettiğiyle bırak. Ama bu basit yol tutarsız sonuçlar verir: bilinmeyen bölge, bilinen bölgenin **bu adımdaki** gürültülü hâline uyar, ileride neye dönüşeceğine değil. RePaint (Lugmayr ve ark. 2022) bunu her adımda birkaç kez geri gidip (yeniden gürültü ekleyip) tekrar ileri gelerek ("yeniden örnekleme") uyumlaştırır; bedeli birkaç kat NFE'dir. Ek x2, kâhin bir modelle bu basit yöntemin nerede başarısız olduğunu gösteriyor. Neyin işe yaramadığını raporlamak da bir araştırma çıktısıdır.

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| GroupNorm | `nn.GroupNorm(4, c)` |
| sıfır başlatma | `nn.init.zeros_(conv.weight); nn.init.zeros_(conv.bias)` |
| EMA | `ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))`; her adım `ema.update_parameters(model)` |
| toplu CFG | `e_u, e_c = net(torch.cat([x, x]), torch.cat([t, t]), torch.cat([null, y])).chunk(2)` |
| eşikleme | `x0 = x0.clamp(-1, 1)` |
| Dice kaybı | `1 - (2 * (p * g).sum() + 1) / (p.sum() + g.sum() + 1)` (p = sigmoid çıktı) |
| eşleştirilmiş bootstrap | `idx = torch.randint(0, n, (B, n)); (b[idx] - a[idx]).mean(1).quantile(torch.tensor([0.025, 0.975]))` |

### Özet {#ozet}

- Difüzyon U-Net'i: GN + SiLU artık bloklar, sıfır başlatılmış çıkış, sinüzoidal zaman embedding'i ($\sin5=-0.959$), en dipte öz-dikkat (256 skor), atlama birleştirmesi (48 kanal).
- Eğitim: etiket bırakma (CFG), EMA (pencere $1/(1-\beta)$; 0.995 → 200; $0.999^{1000}=0.368$).
- Çıkarım: DDIM + CFG 25 adımda 50 NFE; toplu CFG çağrıları yarıya indirir; $\hat x_0$ kırpma.
- Değerlendirme: FID'i gerçek–gerçek tabanla, sınıf doğruluğunu ayrı, yönlendirme ölçeğini tarayarak raporla.
- Segmentasyon: dengesizlikte Dice ($50/70=0.714$), BCE, ağırlıklı BCE, BCE + Dice; hassasiyet–duyarlılık seçimi.
- Karşılaştırma: eşleştirilmiş bootstrap (genişlik ≈ $\sqrt{1-\rho}$ kat); tohumlarla tekrar.
