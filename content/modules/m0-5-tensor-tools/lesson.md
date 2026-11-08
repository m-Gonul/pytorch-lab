### Tensör nedir: veri + onu okuma kuralı {#nedir}

Python'da `x = torch.randn(64, 3, 224, 224)` yazınca bellekte 4 boyutlu bir şey oluşmaz. Bilgisayar belleği **tek boyutludur**: art arda dizilmiş baytlar. Tensör iki parçadan oluşur:

1. **Storage (depo):** elemanların düz, ardışık bir dizisi (burada $64\cdot3\cdot224\cdot224\approx9.6$ milyon float32, ~38.5 MB).
2. **Metadata (üst veri):** bu düz diziyi çok boyutlu bir tablo **gibi okumanın** kuralı — `shape`, `stride`, `storage_offset`, `dtype`, `device`.

:::tanım Shape, stride, offset
- **shape** $(n_0,\dots,n_{d-1})$: her eksenin uzunluğu.
- **stride** $(s_0,\dots,s_{d-1})$: $k$. eksende indisi 1 artırmak için storage'da **kaç eleman** ileri gidileceği.
- **offset** $o$: `x[0, …, 0]`'ın storage'daki konumu.

Bir elemanın adresi:
$$\text{adres}(i_0,\dots,i_{d-1})=o+\sum_{k=0}^{d-1}i_k\,s_k.$$
(PyTorch stride'ı **eleman** cinsinden, NumPy **bayt** cinsinden verir: float64'te NumPy stride'ı PyTorch'unkinin 8 katıdır.)
:::

Bu ayrımın sonucu çok güçlüdür: **aynı storage üzerinde farklı metadata ile farklı tensörler** görebiliriz. Transpoz, dilimleme, boyut ekleme, yayın (broadcasting), kayan pencere — bunların hiçbiri veriyi kopyalamaz; yalnız yeni bir (shape, stride, offset) üçlüsü üretir. Bu işlemler **O(1)**'dir, tensör ne kadar büyük olursa olsun.

<figure class="fig wide"><svg viewBox="0 0 640 210" role="img" aria-label="12 elemanlık storage ve onun üzerine iki görünüm: (3,4) bitişik ve transpozu (4,3)"><g class="sm"><text x="10" y="18">storage (düz bellek)</text><g transform="translate(10,26)"><rect x="0" y="0" width="40" height="26" class="soft-a"/><rect x="40" y="0" width="40" height="26" class="soft-a"/><rect x="80" y="0" width="40" height="26" class="soft-a"/><rect x="120" y="0" width="40" height="26" class="soft-a"/><rect x="160" y="0" width="40" height="26" class="soft-b"/><rect x="200" y="0" width="40" height="26" class="soft-b"/><rect x="240" y="0" width="40" height="26" class="soft-b"/><rect x="280" y="0" width="40" height="26" class="soft-b"/><rect x="320" y="0" width="40" height="26" class="soft-p"/><rect x="360" y="0" width="40" height="26" class="soft-p"/><rect x="400" y="0" width="40" height="26" class="soft-p"/><rect x="440" y="0" width="40" height="26" class="soft-p"/><g class="mute" fill="none"><rect x="0" y="0" width="480" height="26"/></g><text x="20" y="18" text-anchor="middle">0</text><text x="60" y="18" text-anchor="middle">1</text><text x="100" y="18" text-anchor="middle">2</text><text x="140" y="18" text-anchor="middle">3</text><text x="180" y="18" text-anchor="middle">4</text><text x="220" y="18" text-anchor="middle">5</text><text x="260" y="18" text-anchor="middle">6</text><text x="300" y="18" text-anchor="middle">7</text><text x="340" y="18" text-anchor="middle">8</text><text x="380" y="18" text-anchor="middle">9</text><text x="420" y="18" text-anchor="middle">10</text><text x="460" y="18" text-anchor="middle">11</text></g><text x="10" y="82">x: shape (3, 4), stride (4, 1)</text><g transform="translate(10,90)"><rect x="0" y="0" width="160" height="26" class="soft-a"/><rect x="0" y="26" width="160" height="26" class="soft-b"/><rect x="0" y="52" width="160" height="26" class="soft-p"/><g class="mute"><rect x="0" y="0" width="160" height="78" fill="none"/><line x1="40" y1="0" x2="40" y2="78"/><line x1="80" y1="0" x2="80" y2="78"/><line x1="120" y1="0" x2="120" y2="78"/></g><text x="20" y="18" text-anchor="middle">0</text><text x="60" y="18" text-anchor="middle">1</text><text x="100" y="18" text-anchor="middle">2</text><text x="140" y="18" text-anchor="middle">3</text><text x="20" y="44" text-anchor="middle">4</text><text x="60" y="44" text-anchor="middle">5</text><text x="100" y="44" text-anchor="middle">6</text><text x="140" y="44" text-anchor="middle">7</text><text x="20" y="70" text-anchor="middle">8</text><text x="60" y="70" text-anchor="middle">9</text><text x="100" y="70" text-anchor="middle">10</text><text x="140" y="70" text-anchor="middle">11</text></g><text x="300" y="82">x.t(): shape (4, 3), stride (1, 4) — aynı storage</text><g transform="translate(300,90)"><rect x="0" y="0" width="40" height="104" class="soft-a"/><rect x="40" y="0" width="40" height="104" class="soft-b"/><rect x="80" y="0" width="40" height="104" class="soft-p"/><g class="mute"><rect x="0" y="0" width="120" height="104" fill="none"/><line x1="0" y1="26" x2="120" y2="26"/><line x1="0" y1="52" x2="120" y2="52"/><line x1="0" y1="78" x2="120" y2="78"/></g><text x="20" y="18" text-anchor="middle">0</text><text x="60" y="18" text-anchor="middle">4</text><text x="100" y="18" text-anchor="middle">8</text><text x="20" y="44" text-anchor="middle">1</text><text x="60" y="44" text-anchor="middle">5</text><text x="100" y="44" text-anchor="middle">9</text><text x="20" y="70" text-anchor="middle">2</text><text x="60" y="70" text-anchor="middle">6</text><text x="100" y="70" text-anchor="middle">10</text><text x="20" y="96" text-anchor="middle">3</text><text x="60" y="96" text-anchor="middle">7</text><text x="100" y="96" text-anchor="middle">11</text></g><text x="440" y="130" class="tb">satır boyunca okumak</text><text x="440" y="146" class="tb">belleği 4'er atlar</text></g></svg><figcaption>Şekil 1 — Hücrelerdeki sayılar storage adresidir. Transpoz hiçbir elemanı taşımaz; yalnız stride'ları yer değiştirir. Renkler, x'in satırlarının belleğin hangi parçasından geldiğini gösterir.</figcaption></figure>

### Satır-öncelikli düzen ve stride hesabı {#stride}

PyTorch ve NumPy varsayılan olarak **satır-öncelikli** (row-major, C sırası) düzen kullanır: **son eksen** bellekte ardışıktır, ondan önceki eksen bir "satır" uzunluğu kadar atlar ve böyle devam eder. Fortran, MATLAB ve Julia ise **sütun-öncelikli**dir (ilk eksen ardışık).

:::türetim Bitişik (contiguous) stride formülü
Son eksende bir adım = 1 eleman: $s_{d-1}=1$. $k$. eksende bir adım atmak, ondan sonraki eksenlerin oluşturduğu bloğun tamamını atlamak demektir: $s_k=\prod_{j>k}n_j=s_{k+1}\,n_{k+1}$.
Örnek: shape $(2,3,4)$ → $s_2=1$, $s_1=4$, $s_0=12$: stride $(12,4,1)$. Kontrol: `x[1, 2, 3]` → $12+8+3=23$, yani son eleman ✓ (toplam $2\cdot3\cdot4=24$ eleman, adres 0–23).
:::

**Görüntü düzenleri.** (B, C, H, W) = **NCHW** (PyTorch varsayılanı): bir kanalın tüm pikselleri ardışık. **NHWC** (`channels_last`): bir pikselin tüm kanalları ardışık. Mantıksal shape aynı kalır, yalnız stride'lar değişir: `x.to(memory_format=torch.channels_last)` için stride $(HWC,\,1,\,WC,\,C)$. Tensor çekirdekli GPU'larda konvolüsyonlar NHWC'de genelde daha hızlıdır, çünkü kernel'lar kanal boyunca vektörleşir (M7).

### Görünümler: kopyasız işlemler {#gorunumler}

| işlem | yeni shape | yeni stride | offset | not |
|---|---|---|---|---|
| `x.t()`, `x.transpose(0,1)` | eksenler yer değiştirir | stride'lar yer değiştirir | aynı | bitişikliği bozar |
| `x.permute(2,0,1)` | eksenler yeniden sıralanır | stride'lar aynı sırayla | aynı | genel transpoz |
| `x[a:b:c, ...]` | $\lceil(b-a)/c\rceil$ | $c\cdot s_k$ | $o+a\,s_k$ | basit dilimleme |
| `x[i]` | ekseni düşür | o ekseni sil | $o+i\,s_0$ | |
| `x.unsqueeze(k)` | $k$'ya boyu 1 eksen | herhangi (önemsiz) | aynı | `x[:, None]` |
| `x.expand(...)` | 1'lik eksenler büyür | o eksende **0** | aynı | yayın |
| `x.view(...)` | yeni shape | bitişikse hesaplanır | aynı | uyumsuzsa hata |
| `x.unfold(k, w, step)` | $k$'da pencere sayısı + sona $w$ | $(step\cdot s_k,\ \dots,\ s_k)$ | aynı | kayan pencere |
| `x.diagonal()` | köşegen uzunluğu | $s_0+s_1$ | aynı | |

**Kopya yapan işlemler:** `x.contiguous()` (gerekiyorsa), `x.clone()`, **gelişmiş indeksleme** (`x[[0, 2]]`, `x[mask]`, `torch.gather` — sonuç tek bir stride ile ifade edilemez), `x.repeat(...)`, `reshape` (görünüm mümkün değilse), `torch.cat`/`stack`, tüm aritmetik işlemlerin sonuçları.

Aşağıdaki şekilde her görünümün storage'ın hangi hücrelerini hangi sırayla okuduğunu gör; bir hücreye gelince adres formülü açılır. Stride ve offset'i elle değiştirerek kendi görünümünü kur — storage dışına taşarsa uyarı alırsın.

::lab l1

### Bitişiklik, view ve reshape {#contiguous}

:::tanım Bitişik (contiguous)
Bir tensör, elemanları satır-öncelikli sırayla storage'da **ardışık ve boşluksuz** diziliyse bitişiktir; yani stride'ı tam olarak bitişik stride formülüne eşittir (boyu 1 olan eksenlerin stride'ı önemsizdir). `x.is_contiguous()` bunu denetler.
:::

**`view` ile `reshape` farkı.**

- `x.view(shape)` **asla kopyalamaz**. Yeni shape, mevcut stride'larla ifade edilebiliyorsa görünüm döner; edilemiyorsa `RuntimeError: view size is not compatible with input tensor's size and stride` hatası verir.
- `x.reshape(shape)` önce görünüm dener; olmazsa **sessizce kopyalar** (`x.contiguous().view(shape)` gibi). Sonucun orijinalle storage paylaşıp paylaşmadığını bu yüzden garanti etmez.

Neden transpozdan sonra `view(-1)` olmaz? `x.t()`'nin satır-öncelikli okuma sırası adres olarak 0, 4, 8, 1, 5, 9, … dır (Şekil 1). Bu diziyi **tek bir** stride ile (sabit adımla) üretmek imkânsızdır; bir kopya gerekir.

:::yanılgı "reshape ile eksenlerin yerini değiştirebilirim"
`reshape` ve `view` elemanların **satır-öncelikli okuma sırasını korur**, eksenleri yer değiştirmez. (B, C, H, W) bir görüntüyü (B, H, W, C)'ye çevirmek için `x.reshape(B, H, W, C)` **yanlıştır**: hata vermez, ama pikselleri karıştırır (kanal değerleri komşu piksellere dağılır). Doğrusu `x.permute(0, 2, 3, 1)`. Aynı hata attention'da başları ayırırken de yapılır: `(B, T, h·d)` → `(B, T, h, d)` → `permute(0, 2, 1, 3)` → `(B, h, T, d)`; doğrudan `reshape(B, h, T, d)` sessizce yanlış sonuç verir (M4.6).
:::

**Bitişik olmamanın maliyeti.** Çoğu PyTorch kernel'ı keyfi stride'larla çalışabilir, ama: (1) bazıları (`view`, bazı C++/CUDA eklentileri) bitişik ister; (2) atlamalı erişim önbellek satırlarını boşa harcar (M0.2, döngü sırası deneyi); (3) `.contiguous()` çağrısı tam bir kopya ve bellek ayırmasıdır. İleri PyTorch'ta "bu satırda gizli bir kopya var mı?" sorusu sık sorulur; `data_ptr()` ya da `untyped_storage().data_ptr()` karşılaştırması ve profiler (`aten::copy_`, `aten::contiguous`) cevabı verir.

### Yayın (broadcasting) = stride 0 {#yayin}

M0.2'deki yayın kuralları bellekte şöyle uygulanır: boyu 1 olan eksen, istenen uzunluğa **stride 0 ile** "uzatılır". `b.shape == (10,)` bir bias'ı (64, 10)'luk çıktıya eklerken PyTorch `b.expand(64, 10)` görünümünü kullanır: shape (64, 10), stride (0, 1). 64 satırın hepsi aynı 10 elemanı okur; 640 elemanlık bir kopya hiç oluşmaz.

| | `expand` | `repeat` |
|---|---|---|
| bellek | kopya yok (stride 0) | gerçekten kopyalar |
| yazma | **tehlikeli**: birçok indis aynı hücreye işaret eder; PyTorch çoğu yerinde (in-place) işlemi reddeder | güvenli |
| ne zaman | salt okunur yayın | değişecek bağımsız kopyalar gerekiyorsa |

### Kayan pencere ve as_strided {#pencere}

Bir sinyal üzerinde $w$ uzunluklu pencereler (hareketli ortalama, 1-B konvolüsyon): $x$ (n,) için pencere matrisi $W_{ij}=x_{i\cdot\text{step}+j}$, shape $(\lfloor(n-w)/\text{step}\rfloor+1,\ w)$. Adres formülüne bakınca bunun bir görünüm olduğu görülür: **stride $(\text{step}\cdot s,\ s)$, offset aynı**. `x.unfold(0, w, step)` tam olarak bunu döndürür; step < w ise pencereler **çakışır** (aynı eleman birden çok satırda görünür) ve hiç kopya yapılmaz.

Ardından `W.mean(1)` hareketli ortalamadır ($O(nw)$ iş, döngü yok); `W @ k` ise 1-B konvolüsyondur — **im2col** fikri: konvolüsyonu pencerelerin matris çarpımına çevirmek (M4.1). 2-B'de `F.unfold` aynı şeyi görüntü yamaları için yapar.

:::uyarı as_strided bellek güvenliği sağlamaz
`torch.as_strided(x, size, stride, offset)` ve NumPy `np.lib.stride_tricks.as_strided` istediğin metadata'yı **denetlemeden** kurar. Bir adres storage'ın dışına düşerse rastgele bellek okursun (NumPy'da çöpe benzeyen sayılar ya da çökme). Çakışan bir görünüme yazarsan sonuç tanımsızdır. Mümkünse güvenli sarmalayıcıları kullan: `unfold`, `np.lib.stride_tricks.sliding_window_view`.
:::

### Paylaşılan bellek, yerinde işlemler ve autograd {#aliasing}

Görünümler storage paylaştığı için birine **yazmak** diğerlerini de değiştirir:

```python
x = torch.zeros(3, 4)
r = x[1]          # görünüm
r += 5            # x[1] de 5 oldu
y = x.t()
y[0, 2] = -1      # x[2, 0] = -1
```

Bu bir hata değil özelliktir (bellek tasarrufu, yerinde güncelleme), ama yanlışlıkla olursa bulması zor hatalar üretir. Kural: bağımsız bir kopya istiyorsan `clone()`.

**Autograd ile etkileşim.** Geri geçiş, ileri geçişte kaydedilen değerlere ihtiyaç duyar (M0.3). Bir tensör kaydedildikten sonra yerinde değiştirilirse PyTorch bunu **sürüm sayacı** (version counter) ile fark eder ve geri geçişte `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` hatası verir. Ayrıca:

- `x.detach()` aynı storage'ı paylaşan ama grafiğe bağlı olmayan bir tensör döndürür (kopya değil!). Ona yazmak orijinali de değiştirir.
- `x.detach().clone()` hem grafikten kopar hem kopyalar.
- NumPy ile köprü: `t.numpy()` ve `torch.from_numpy(a)` (CPU'da) **belleği paylaşır**.

### dtype ve bellek {#dtype}

| dtype | bayt | aralık / hassasiyet | kullanım |
|---|---|---|---|
| `float32` | 4 | ~7 ondalık basamak, ε ≈ 1.2·10⁻⁷ | varsayılan |
| `float64` | 8 | ~16 basamak | gradyan kontrolü, bilimsel hesap |
| `float16` | 2 | ~3 basamak, en büyük 65504 | karışık hassasiyet (taşma riski) |
| `bfloat16` | 2 | float32 aralığı, ~2–3 basamak | büyük model eğitimi |
| `int64` (`long`) | 8 | tam sayı | indisler, sınıf etiketleri |
| `uint8` | 1 | 0–255 | ham görüntüler |
| `bool` | 1 | | maskeler |

Bellek = eleman sayısı × eleman boyu: 7 milyar parametreli bir modelin ağırlıkları bf16'da ~14 GB, Adam'ın iki moment tensörü float32'de +56 GB. `x.element_size()`, `x.nelement()`, `x.untyped_storage().nbytes()` bu hesabı yapar. **Dikkat:** küçük bir dilimin görünümü, tüm storage'ı canlı tutar; büyük bir tensörün küçük bir parçasını saklamak istiyorsan `clone()` et ki büyük blok serbest kalsın.

### Benchmark disiplini: neden saf zaman ölçümü yalan söyler {#benchmark}

"Hangisi daha hızlı?" sorusunun cevabı ölçümdür, ama naif ölçüm (`t=time.time(); f(); print(time.time()-t)`) birçok şekilde yanıltır:

1. **Isınma:** ilk çağrı bellek ayırıcıyı, CUDA bağlamını, cuDNN algoritma seçimini, `torch.compile` derlemesini, işletim sistemi sayfa hatalarını tetikler. İlk ölçümler **atılır**.
2. **Gürültü:** işletim sistemi kesintileri, başka süreçler, CPU frekans değişimi (turbo, ısınma), önbellek durumu. Tek ölçüm anlamsızdır; **çok tekrar** gerekir.
3. **Çarpık dağılım:** gürültü süreyi yalnız **uzatır**. Ortalama birkaç yavaş ölçümle yukarı çekilir; **medyan** dayanıklıdır. Yayılımı da (çeyrekler arası aralık, IQR) raporla.
4. **Eşzamansızlık (GPU):** CUDA işlemleri kuyruğa konur, Python hemen döner. Senkronizasyon (`torch.cuda.synchronize()`) olmadan ölçülen şey yalnız başlatmadır.
5. **Ölü kod ve önbellek:** sonucu kullanılmayan hesap derleyicide atılabilir; aynı küçük girdiyi tekrar tekrar ölçmek hep önbellekte çalışır (gerçek iş yükünde öyle olmayabilir).
6. **Zamanlayıcı çözünürlüğü:** `time.time()` yerine `time.perf_counter()`; çok kısa işlemleri bir döngüde çok kez çalıştırıp toplamı böl.

Aşağıdaki benzetimde bu etkileri tek tek aç kapat ve özetlerin (ortalama, medyan, kırpılmış ortalama, en küçük) gerçek süreye ne kadar yakın olduğunu 200 bağımsız deneyle karşılaştır.

::lab l2

**`torch.utils.benchmark` bunları senin yerine yapar:**

```python
from torch.utils import benchmark
t = benchmark.Timer(stmt="x.unfold(0, w, 1).mean(1)",
                    globals={"x": x, "w": 64}, num_threads=1,
                    label="hareketli ortalama", sub_label="unfold")
m = t.blocked_autorange(min_run_time=0.5)   # ısınma + yeterli tekrar, otomatik
print(m.median, m.iqr)                      # saniye; medyan ve çeyrekler arası aralık
```

`Timer` GPU'da senkronizasyonu kendisi yapar, iş parçacığı sayısını sabitler (karşılaştırmalar adil olsun), ölçümleri bloklar hâlinde toplar ve `Compare` ile tablo hâlinde gösterir. Ne kadar süre harcandığını ("nerede?") bulmak için ise **profiler** kullanılır: `torch.profiler.profile` hangi operatörün ne kadar sürdüğünü, kaç kopya yapıldığını ve bellek ayırmalarını gösterir (M7).

**Neyi raporla:** donanım ve sürümler, girdi boyutları, dtype, iş parçacığı sayısı, medyan ± IQR, tekrar sayısı. "X, Y'den 3 kat hızlı" iddiası bu bilgiler olmadan tekrarlanamaz.

:::ayrıntı Hesap mı, bellek mi? Kaba bir sınır
Bir işlemin süresi kabaca $\max\big(\frac{\text{FLOP}}{\text{tepe FLOP/s}},\ \frac{\text{taşınan bayt}}{\text{bellek bant genişliği}}\big)$'dir (roofline modeli). Eleman eleman işlemler (toplama, ReLU) her eleman için birkaç FLOP yapıp 8–12 bayt taşır: **bellek sınırlı**dır; hızlandırmanın yolu daha az okuma/yazmadır (işlemleri birleştirmek, `torch.compile`'ın kernel füzyonu). Büyük matris çarpımı ise eleman başına çok FLOP yapar: **hesap sınırlı**dır (M0.2). Bir ölçümü yorumlarken önce hangisi olduğunu sor: bellek sınırlı bir işlemi hesap açısından optimize etmek hiçbir şey kazandırmaz.
:::

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| storage + metadata | tensör = düz bellek + (shape, stride, offset, dtype, device) |
| adres | $o+\sum_ki_ks_k$ |
| bitişik stride | $s_k=\prod_{j>k}n_j$, son eksen 1 |
| görünüm (kopya yok) | transpoz, permute, dilim, unsqueeze, expand (stride 0), unfold, view |
| kopya | contiguous, clone, gelişmiş indeksleme, repeat, (bazen) reshape |
| view vs reshape | view asla kopyalamaz (gerekirse hata), reshape gerekirse kopyalar |
| reshape ≠ permute | reshape okuma sırasını korur; eksen değiştirmek için permute |
| ölçüm | ısınmayı at, çok tekrar, medyan + IQR, GPU'da senkronize et, `torch.utils.benchmark` |

**Kaynaklar:** PyTorch belgeleri, "Tensor Views" ve `torch.utils.benchmark` tarifi; Edward Z. Yang, "PyTorch internals" (2019; storage/stride modelinin en iyi anlatımı); NumPy, "Internal memory layout of an ndarray" ve `as_strided` belgesi; Williams, Waterman ve Patterson (2009), "Roofline: an insightful visual performance model"; PyTorch, "Channels Last Memory Format" eğitimi.
