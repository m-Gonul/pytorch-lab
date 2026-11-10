### Neden küçültmek? {#neden}

Eğitilmiş bir modeli çalıştırmanın (çıkarım) maliyeti çoğu zaman **bellek** tarafından belirlenir: ağırlıkların GPU'ya, telefona ya da tarayıcıya sığması ve her adımda bellekten okunması gerekir. Dil modelinin token token üretiminde her token için bütün ağırlıklar bir kez okunur ve işlem bellek bağımlıdır (M6.5'teki matris–vektör çarpımı). Ağırlıkları 16 bitten 4 bite indirmek bu durumda hızı da yaklaşık 4 kat artırır. İki temel araç vardır: **nicemleme** (quantization: her sayıyı daha az bitle temsil et) ve **budama** (pruning: sayıların bir kısmını sıfırla ve atla).

### Simetrik ve afin nicemleme {#nicemleme}

Nicemleme gerçek sayıları bir tamsayı ızgarasına eşler: $\hat x=s\cdot(q-z)$, $q$ tamsayı.
- **Simetrik**: $z=0$, $s=\max|x|/(2^{b-1}-1)$, $q=\mathrm{clamp}(\mathrm{round}(x/s),-2^{b-1},2^{b-1}-1)$. 0 tam olarak 0'a düşer. Ağırlıklar gibi sıfır ortalamalı veriye uygundur.
- **Afin** (asimetrik): $s=(\max-\min)/(2^b-1)$, $z=\mathrm{round}(-\min/s)$. Aralık veriye kaydırılır; 0 yine tam temsil edilir (z koduyla). ReLU sonrası aktivasyonlar gibi tek işaretli veriye uygundur. Simetrik şema bu veride kodların yarısını hiç kullanmaz.

<figure class="fig wide"><svg viewBox="0 0 640 220" role="img" aria-label="Simetrik int8'de 0.5, eksi 1.2 ve 2.54 değerleri 25, eksi 60 ve 127 kodlarına; afin uint8'de eksi 1 ile 3 aralığı 0 ile 255'e, sıfır 64'e eşlenir"><text x="10" y="18" class="ta">Simetrik int8: s = max|x| / 127 = 2.54 / 127 = 0.02</text><path d="M60,50 L600,50" class="mute"/><path d="M66.2,45 L66.2,55" class="mute"/><text x="66.2" y="70" text-anchor="middle" class="sm">-2.54</text><path d="M226.2,45 L226.2,55" class="mute"/><text x="226.2" y="70" text-anchor="middle" class="sm">-1</text><path d="M330.0,45 L330.0,55" class="mute"/><text x="330.0" y="70" text-anchor="middle" class="sm">0</text><path d="M433.8,45 L433.8,55" class="mute"/><text x="433.8" y="70" text-anchor="middle" class="sm">1</text><path d="M593.8,45 L593.8,55" class="mute"/><text x="593.8" y="70" text-anchor="middle" class="sm">2.54</text><text x="66.2" y="38" text-anchor="middle" class="sm">−127</text><text x="593.8" y="38" text-anchor="middle" class="sm">127</text><circle cx="381.9" cy="50" r="5" class="ca-f"/><text x="381.9" y="86" text-anchor="middle" class="ta">q = 25</text><circle cx="205.4" cy="50" r="5" class="ca-f"/><text x="205.4" y="86" text-anchor="middle" class="ta">q = -60</text><circle cx="593.8" cy="50" r="5" class="ca-f"/><text x="593.8" y="86" text-anchor="middle" class="ta">q = 127</text><text x="10" y="116" class="tb">Afin uint8, aralık [−1, 3]: s = 4 / 255, z = round(1 / s) = 64</text><path d="M60,146 L600,146" class="mute"/><path d="M60.0,141 L60.0,151" class="mute"/><text x="60.0" y="166" text-anchor="middle" class="sm">-1</text><text x="60.0" y="134" text-anchor="middle" class="sm">0</text><path d="M195.0,141 L195.0,151" class="mute"/><text x="195.0" y="166" text-anchor="middle" class="sm">0</text><text x="195.0" y="134" text-anchor="middle" class="sm">64</text><path d="M262.5,141 L262.5,151" class="mute"/><text x="262.5" y="166" text-anchor="middle" class="sm">0.5</text><text x="262.5" y="134" text-anchor="middle" class="sm">96</text><path d="M600.0,141 L600.0,151" class="mute"/><text x="600.0" y="166" text-anchor="middle" class="sm">3</text><text x="600.0" y="134" text-anchor="middle" class="sm">255</text><circle cx="262.5" cy="146" r="5" class="cb-f"/><text x="270.5" y="184" class="tb">x = 0.5 → q = 32 + 64 = 96 → (96 − 64)·s = 0.50196</text><text x="10" y="210" class="sm">Simetrikte 0 tam 0’a düşer; afin şema aralığı veriye kaydırır ve 0’ı z koduyla tam temsil eder.</text></svg><figcaption>Şekil 1 — Modülün iki el hesabı. Simetrik şemada ölçek en büyük mutlak değere göre seçilir; afin şema bir zero-point ile aralığı tam kullanır.</figcaption></figure>

**El hesapları.** Simetrik int8, $\mathbf x=(0.5,-1.2,2.54)$: $s=2.54/127=\mathbf{0.02}$, $\mathbf q=(25,\mathbf{-60},\mathbf{127})$. Afin uint8, aralık $[-1,3]$: $s=4/255$, $z=\mathrm{round}(1/s)=\mathbf{64}$; $x=0.5$ için $q=\mathrm{round}(31.875)+64=96$ ve $\hat x=(96-64)\cdot4/255=\mathbf{0.50196}$.

:::türetim Nicemleme hatası ve bit başına kazanç
Yuvarlama hatası $\epsilon=\hat x-x$, $[-s/2,s/2]$'de yaklaşık düzgündür (kırpma yoksa): $\mathbb E[\epsilon^2]=s^2/12$. $s\propto\text{aralık}/2^b$ olduğu için **her ek bit hatanın std'sini yarıya** indirir (≈ 6 dB). Simetrik şemada aralık $2\max|x|$'tir: tek bir aykırı değer $\max|x|$'i, dolayısıyla bütün değerlerin hatasını büyütür. Ölçeği küçük bloklarda seçmek bu yüzden önemlidir.
:::

### Ölçek kapsamı: tensör, kanal, grup {#kapsam}

Ölçek hangi değer kümesi için seçilir?
- **Tensör başına**: tek bir $s$. Satırlar (çıkış kanalları) farklı ölçekliyse küçük satırlar çok kaba bir ızgarada kalır.
- **Kanal başına**: her çıkış kanalının (satırın) kendi ölçeği. Ağırlıklarda standarttır; matris çarpımından sonra ölçek satır başına çarpılarak geri uygulanır (aşama c, d).
- **Grup başına**: her satır $g$'lik gruplara bölünür (tipik $g=32$–$128$), her grubun kendi ölçeği. Bir aykırı değer yalnız kendi grubunu bozar. LLM'lerin 4 bitlik ağırlık nicemlemesinde (GPTQ, AWQ, llama.cpp) standarttır. Bedeli ölçeklerin bitidir: fp16 ölçek ile bit/ağırlık $=b+16/g$.

Aşağıdaki deneyde 4 bitte, %0.2 aykırı değer (× 10) varken ölçülen ağırlık hatası: tensör başına 0.69, kanal başına 0.27, 64'lük grup başına 0.16 (4.25 bit/ağırlık).

::lab l1

### Tamsayı matris çarpımı {#int8mm}

Hem ağırlıklar ($W\approx s_w\,Q_w$) hem girdiler ($x\approx s_x\,q_x$) nicemlenirse matris çarpımı tamamen tamsayıda yapılabilir:
$$y=Wx\approx s_w\,s_x\,(Q_w\,q_x).$$
$Q_wq_x$, int8 × int8 çarpımlarının **int32** birikimidir (255 · 255 · binlerce terim int16'ya sığmaz). Kanal başına ölçeklerde $s_w$ satır başına bir vektördür ve çarpımdan sonra satırlara uygulanır. Tamsayı çarpımı donanımda fp16'dan daha hızlı ve daha az enerji harcayan birimlerde yapılır (Jacob ve ark. 2018). Aşama c bunu NumPy'da yazıyor.

**Aktivasyonlardaki aykırı değerler.** Büyük transformer'larda aktivasyonların birkaç **özellik boyutu** sistematik olarak çok büyük değerler alır (Dettmers ve ark. 2022). Bu boyutlar tensör başına int8 ölçeği bozar ve doğruluk ~6.7 milyar parametrenin üstünde çöker. LLM.int8() bu birkaç boyutu fp16'da, geri kalanını int8'de çarparak çözer. SmoothQuant ise aykırılığı bir ölçekle aktivasyondan ağırlığa aktarır. Bu yüzden LLM'lerde en yaygın yaklaşım **ağırlık-yalnız** nicemlemedir: ağırlıklar int4/int8 saklanır, çarpımdan hemen önce fp16'ya açılır.

### Grup bazlı int4 ve paketleme {#int4}

<figure class="fig wide"><svg viewBox="0 0 640 204" role="img" aria-label="Grup bazlı int4: bir satır 64'lük gruplara bölünür, her grubun fp16 ölçeği vardır; iki dört bitlik değer bir bayta paketlenir"><text x="10" y="18" class="sm">Bir satır (in = 256), g = 64’lük gruplar; her grubun kendi fp16 ölçeği s = max|w| / 7</text><rect x="10" y="26" width="150" height="26" rx="3" class="soft-a" stroke="var(--line)"/><text x="85" y="43" text-anchor="middle" class="sm">grup 0: 64 × int4</text><rect x="60" y="58" width="50" height="20" rx="3" class="soft-b" stroke="var(--line)"/><text x="85" y="72" text-anchor="middle" class="sm">s0 (fp16)</text><rect x="165" y="26" width="150" height="26" rx="3" class="soft-a" stroke="var(--line)"/><text x="240" y="43" text-anchor="middle" class="sm">grup 1: 64 × int4</text><rect x="215" y="58" width="50" height="20" rx="3" class="soft-b" stroke="var(--line)"/><text x="240" y="72" text-anchor="middle" class="sm">s1 (fp16)</text><rect x="320" y="26" width="150" height="26" rx="3" class="soft-a" stroke="var(--line)"/><text x="395" y="43" text-anchor="middle" class="sm">grup 2: 64 × int4</text><rect x="370" y="58" width="50" height="20" rx="3" class="soft-b" stroke="var(--line)"/><text x="395" y="72" text-anchor="middle" class="sm">s2 (fp16)</text><rect x="475" y="26" width="150" height="26" rx="3" class="soft-a" stroke="var(--line)"/><text x="550" y="43" text-anchor="middle" class="sm">grup 3: 64 × int4</text><rect x="525" y="58" width="50" height="20" rx="3" class="soft-b" stroke="var(--line)"/><text x="550" y="72" text-anchor="middle" class="sm">s3 (fp16)</text><text x="10" y="102" class="sm">Bit/ağırlık = 4 + 16 / 64 = 4.25 (fp16’ya göre 3.8 kat küçük). Bir aykırı değer yalnız kendi grubunun ölçeğini bozar.</text><text x="10" y="132" class="tb">Paketleme: iki int4 bir bayta</text><rect x="200" y="142" width="160" height="30" class="soft-p" stroke="var(--line)"/><text x="280" y="161" text-anchor="middle" class="sm">0101 (q₁ &lt;&lt; 4)</text><rect x="360" y="142" width="160" height="30" class="soft-a" stroke="var(--line)"/><text x="440" y="161" text-anchor="middle" class="sm">1101 (q₀ &amp; 0xF)</text><text x="10" y="161" class="sm">q₀ = −3, q₁ = 5 →</text><text x="530" y="161" class="sm">= 0x5D</text><text x="10" y="194" class="sm">Açarken alt dört bit 1101 = 13 > 7 olduğu için 13 − 16 = −3 (dört bitlik ikiye tümleyen); üst dört bit 0101 = 5.</text></svg><figcaption>Şekil 2 — LLM'lerde ağırlık-yalnız int4 nicemleme (aşama e). Gruplar küçüldükçe hata azalır ama ölçeklerin bit payı artar.</figcaption></figure>

4 bitlik değerlerin donanımda kendi tipi yoktur; iki değer bir bayta paketlenir: `(q0 & 0xF) | ((q1 & 0xF) << 4)`. Açarken alt ve üst dört bit ayrılır ve dört bitlik ikiye tümleyen geri işaretlenir: $v>7$ ise $v-16$. Aşama e bunu döngüsüz yazıyor.

**Eğitim sonrası mı, eğitimde mi?** Ağırlıkları eğitimden sonra doğrudan yuvarlamak (RTN, round-to-nearest) 8 bitte genellikle yeterlidir. 4 bitte doğruluk düşer. **GPTQ** (Frantar ve ark. 2022) ağırlıkları sütun sütun nicemler ve her sütunun yuvarlama hatasını, ikinci dereceden bir yaklaşımla (girdilerin Hessian'ı) henüz nicemlenmemiş sütunlara dağıtarak telafi eder. **Nicemleme farkında eğitim** (QAT) ise ileri geçişte nicemleyip geri geçişte STE (M6.2) kullanarak modelin nicemlemeye uyum sağlamasını öğretir.

### Budama {#budama}

Ağırlıkların önemli bir kısmı çok küçüktür; onları sıfırlamak modelin çıktısını az değiştirir. **Büyüklüğe göre budama** (Han ve ark. 2015) en küçük $|w|$'leri atar. El hesabı: $\mathbf w=(0.1,-2,0.3,1.5)$, %50 budama: en küçük iki ($0.1$, $0.3$) sıfırlanır, **2** sıfır olmayan ağırlık kalır ve $w_3=\mathbf0$.

:::türetim Neden büyüklük? Çıktı hatası ve atılan enerji
Girdiler bağımsız, birim varyanslı ise budanmış $\tilde W$ ile çıktı hatası $\mathbb E\|(W-\tilde W)x\|^2=\|W-\tilde W\|_F^2$'dir: atılan ağırlıkların kareleri toplamı. Belirli sayıda ağırlık atılacaksa bu toplamı en küçükleyen seçim, **en küçük** $|w|$'leri atmaktır. Ağır kuyruklu (Laplace benzeri) dağılımlarda enerjinin çoğu az sayıda büyük ağırlıkta olduğu için yüksek seyreklikte bile kayıp küçüktür. Gerçek girdiler bağımsız değildir; Optimal Brain Damage/Surgeon ve SparseGPT, girdilerin Hessian'ını hesaba katarak daha iyi seçer.
:::

**Global mi, katman başına mı?** Global budama bütün katmanlar için tek bir eşik kullanır: küçük ölçekli katmanlar orantısız budanabilir. Katman (ya da satır) başına budama her birinde aynı oranı uygular.

**Yapısız ve yapılı seyreklik.** Rastgele dağılmış sıfırlar matrisi küçültmez ve donanımda düzensiz bellek erişimi yüzünden hızlandırması zordur (genellikle %90+ seyreklikte işe yarar). Yapılı seyreklik hızlandırılabilir:
- **2:4**: her 4'lük grupta en fazla 2 sıfırdan farklı değer. NVIDIA Ampere ve sonrası tensör çekirdekleri bunu ≈ 2 kat matmul hızıyla çalıştırır.
- **Nöron/kanal/başlık budama**: bütün satırları (nöronları) ya da dikkat başlıklarını atmak matrisi gerçekten küçültür ama en çok hatayı getirir.

Budamadan sonra kısa bir ince ayar kaybın çoğunu geri kazandırır. Kademeli budama (seyrekliği eğitim boyunca artırmak) tek seferliğe göre daha iyidir. Aşağıdaki deneyde %50 seyreklikte, Laplace ağırlıklarla çıktı hatası yapısız budamada 0.16, 2:4'te 0.29, nöron budamada 0.50.

::lab l2

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| kanal başına int8 (elle) | `s = W.abs().amax(1) / 127; q = (W / s[:, None]).round().clamp(-128, 127).to(torch.int8)` |
| geri açma | `W_hat = q.float() * s[:, None]` |
| int4 paketleme | `p = (q[..., 0::2] & 0xF) | ((q[..., 1::2] & 0xF) << 4)` (uint8) |
| kütüphane | `torchao.quantization.quantize_(model, Int8WeightOnlyConfig())` ya da `Int4WeightOnlyConfig(group_size=128)` |
| büyüklüğe göre budama | `prune.l1_unstructured(module, "weight", amount=0.5)` |
| global budama | `prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.5)` |
| maskeyi kalıcı yapmak | `prune.remove(module, "weight")` |
| 2:4 | `torch.sparse.to_sparse_semi_structured(W)` (desteklenen GPU'larda) |

### Özet {#ozet}

- Simetrik: $s=\max|x|/(2^{b-1}-1)$ (el hesabı $s=0.02$, $q=(25,-60,127)$); afin: zero-point ile aralığı veriye kaydırır ($z=64$, 0.5 → 0.50196).
- Yuvarlama hatası $\approx s^2/12$; her bit hatayı yarıya indirir; aykırı değerler ölçeği bozar.
- Ölçek kapsamı: tensör < kanal < grup; grup başına bit maliyeti $16/g$ (g = 64: 4.25 bit/ağırlık).
- int8 × int8 → int32 birikim, ölçeklerle geri dönüş; aktivasyon aykırılıkları (LLM.int8, SmoothQuant); LLM'lerde ağırlık-yalnız int4 (GPTQ, AWQ), paketleme ve işaret geri kazanımı.
- Budama: çıktı hatası atılan enerjiyle orantılı, bu yüzden en küçük $|w|$'ler (el hesabı 2 ağırlık kalır, $w_3=0$). Yapısız en az hata, 2:4 donanım hızı, nöron budama gerçek küçülme; ardından ince ayar.
