### Bellek nereye gidiyor? {#kalemler}

"CUDA out of memory" hatası, derin öğrenmede en sık karşılaşılan pratik sınırdır. Çözmek için önce eğitim belleğinin nereye gittiğini saymak gerekir. Dört büyük kalem vardır:
1. **Ağırlıklar**: $P$ parametre × tip boyu.
2. **Gradyanlar**: ağırlıklarla aynı boyda (her parametre için bir `.grad`).
3. **Optimizasyon durumu**: Adam her parametre için iki moment ($m$, $v$) tutar; SGD + momentum bir tane, düz SGD hiç.
4. **Aktivasyonlar**: geri geçiş için saklanan tensörler (M6.1). Batch boyu ve dizi uzunluğuyla büyür.

Bunlara geçici tamponlar (ör. matris çarpımının çalışma alanı, iletişim tamponları) ve bellek ayırıcının parçalanması (fragmentation) eklenir.

**El hesabı.** GPT-2 small, $124\,439\,808$ parametre, fp32 Adam: ağırlık 4 + gradyan 4 + iki moment 8 = parametre başına **16 bayt**, toplam $\mathbf{1\,991\,036\,928}$ bayt ≈ 1.99 GB. Karışık hassasiyette (bf16 ağırlık ve gradyan + fp32 ana kopya + iki fp32 moment) da toplam $2+2+4+4+4=16$ bayttır (ZeRO makalesindeki hesap). Karışık hassasiyet bu kalemleri küçültmez; kazancı hız ve aktivasyon belleğidir.

<figure class="fig wide"><svg viewBox="0 0 640 212" role="img" aria-label="GPT-2 small eğitiminde bellek: ağırlık, gradyan ve Adam durumu 1.99 GB; aktivasyonlar 8.6 GB; tam yeniden hesapla aktivasyonlar 0.15 GB'a iner"><text x="112" y="41" text-anchor="end" class="sm">checkpoint'siz</text><rect x="120.0" y="20" width="18.6" height="32" class="soft-a" stroke="var(--line)"/><rect x="138.6" y="20" width="18.6" height="32" class="soft-a" stroke="var(--line)"/><rect x="157.2" y="20" width="37.2" height="32" class="soft-p" stroke="var(--line)"/><rect x="194.4" y="20" width="96.0" height="32" class="soft-b" stroke="var(--line)"/><rect x="290.4" y="20" width="225.8" height="32" class="soft-b" stroke="var(--line)"/><text x="524.2" y="41" class="ink">10.60 GB</text><text x="112" y="97" text-anchor="end" class="sm">tam yeniden hesap</text><rect x="120.0" y="76" width="18.6" height="32" class="soft-a" stroke="var(--line)"/><rect x="138.6" y="76" width="18.6" height="32" class="soft-a" stroke="var(--line)"/><rect x="157.2" y="76" width="37.2" height="32" class="soft-p" stroke="var(--line)"/><rect x="194.4" y="76" width="5.6" height="32" class="soft-b" stroke="var(--line)"/><text x="208.1" y="97" class="ink">2.14 GB</text><rect x="10" y="128" width="14" height="12" class="soft-a" stroke="var(--line)"/><text x="30" y="139" class="sm">ağırlık, gradyan (her biri 0.50 GB)</text><rect x="220" y="128" width="14" height="12" class="soft-p" stroke="var(--line)"/><text x="240" y="139" class="sm">Adam m, v (1.00 GB)</text><rect x="430" y="128" width="14" height="12" class="soft-b" stroke="var(--line)"/><text x="450" y="139" class="sm">aktivasyonlar</text><text x="10" y="166" class="sm">GPT-2 small (124 439 808 parametre), batch 8 × 1024 token. Ağırlık + gradyan + Adam (fp32): 16 bayt/parametre.</text><text x="10" y="182" class="sm">Aktivasyon (16 bit, Korthikanti ve ark. 2022): katman başına sbh(34 + 5as/h) bayt, 12 katmanda 8.61 GB;</text><text x="10" y="198" class="sm">bunun 6.04 GB’ı dikkat skorları ve softmax (s² ile büyür). Tam yeniden hesapta yalnız katman girdileri: 0.15 GB.</text></svg><figcaption>Şekil 1 — Eğitim belleğinin kalemleri, hesapla. Uzun dizilerde ve büyük batch'lerde aktivasyonlar baskındır; checkpointing yalnız bu kalemi küçültür (bedeli fazladan ileri geçiş), parametre tarafını ZeRO ve karışık hassasiyet hedefler.</figcaption></figure>

:::türetim Transformer katmanının aktivasyon belleği
Korthikanti ve ark. (2022), 16 bitlik aktivasyonlarla bir transformer katmanının sakladığı baytı sayar ($s$: dizi uzunluğu, $b$: batch, $h$: gizli boyut, $a$: dikkat başı):
$$\text{katman başına}=sbh\Big(34+\frac{5as}{h}\Big)\ \text{bayt}.$$
- $34\,sbh$: doğrusal katmanların girdileri, LayerNorm girdileri, GELU girdisi, dropout maskeleri (1 bayt) gibi dizi uzunluğuyla **doğrusal** büyüyen terimler (M6.1'deki sayımın tam transformer hâli).
- $5as^2b$: dikkat skorları $QK^\top$, softmax çıktısı ve dropout maskesi. Dizi uzunluğuyla **karesel** büyür.

GPT-2 small için ($s=1024$, $b=8$, $h=768$, $a=12$, 12 katman) $2.57+6.04=8.61$ GB: parametre tarafının 4 katından fazla, ve çoğu dikkat matrisi. FlashAttention bu yüzden aktivasyon belleğini de küçültür: $s\times s$ matrisi hiç saklamaz, geri geçişte bloklar hâlinde yeniden hesaplar.
:::

### Aktivasyon checkpointing {#checkpoint}

Aktivasyon belleğini azaltmanın genel yolu, bazılarını saklamayıp geri geçişte **yeniden hesaplamaktır** (Chen ve ark. 2016; "gradient checkpointing" ya da "rematerialization" de denir). $n$ katmanlı bir zinciri $k$'lık segmentlere bölelim. İleri geçişte yalnız segment sonlarını sakla. Geri geçişte segmentleri sondan başa işle: bir segmentin iç çıktılarını, saklanan başlangıcından yeniden hesapla, sonra o segmentte geri git.

:::türetim √n kuralı
Her katman çıktısı 1 birim olsun. Segment $j$ işlenirken bellekte segment $0..j$'nin sonları ve segment $j$'nin yeniden hesaplanan $k-1$ iç çıktısı vardır. Tepe, son segmentte gelir:
$$M(k)\approx\frac nk+k-1 .$$
Türev $-n/k^2+1=0$ ⇒ $k^*=\sqrt n$, $M(k^*)\approx2\sqrt n$. Yeniden hesaplanan katman sayısı $n-n/k\approx n$: bedel en fazla **bir ileri geçiş daha** (geri geçiş zaten ileri geçişin ≈ 2 katı olduğu için toplam süre ≈ %33 artar). Bellek $O(n)$'den $O(\sqrt n)$'ye iner. Segmentleri özyinelemeli olarak yeniden bölmek belleği $O(\log n)$'ye indirir, hesabı $O(n\log n)$'ye çıkarır.
:::

**El hesabı.** 16 katman, $k=4$: checkpoint'siz geri geçişin başında **16** çıktı saklıdır. Checkpoint'li tepe, son segment işlenirken gelir: 4 segment sonu + 3 yeniden hesaplanan = **7**. Toplam yeniden hesap $4\cdot3=\mathbf{12}$ katman. B = 64, genişlik 256, fp32'de bir katman çıktısı $64\cdot256\cdot4=\mathbf{65\,536}$ bayt.

Aşağıdaki deney bellek zaman çizelgesini adım adım çiziyor (ileri geçişte birikim, geri geçişte her segmentin yeniden hesaplanması ve serbest bırakılması) ve tepeyi $k$'ya göre tarıyor.

::lab l1

**PyTorch'ta.** `torch.utils.checkpoint.checkpoint(fn, *girdiler, use_reentrant=False)` `fn`'i ileri geçişte autograd'a kayıt yapmadan çalıştırır; yalnız girdileri saklar ve geri geçişte `fn`'i yeniden çalıştırır. 16 bloklu bir zincirde k = 4 ile saklanan bayt, katman çıktıları için 4 segment girdisine (4 × 65 536) iner; bunu `saved_tensors_hooks` ile ölçtük. Aşama d aynısını yapıyor. Ayrıntılar:
- `use_reentrant=False` önerilen gerçekleştirimdir: saved_tensors_hooks üzerine kuruludur, girdilerden hiçbiri gradyan istemese bile çalışır ve iç içe checkpoint'i destekler.
- **Rastgelelik**: dropout'un yeniden hesapta aynı maskeyi üretmesi için RNG durumu saklanıp geri yüklenir (`preserve_rng_state=True`, varsayılan).
- **Yan etkiler** yeniden hesapta tekrarlanır. Ölçtük: checkpoint içindeki bir BatchNorm'un `num_batches_tracked` sayacı ileri geçişten sonra 1, geri geçişten sonra **2**. Koşu istatistikleri iki kez güncellenir. BN'li bloklarda dikkat gerekir.
- **Seçici yeniden hesaplama**: her şeyi değil, ucuz ama büyük olanları yeniden hesapla. Korthikanti ve ark. yalnız dikkatin $5as^2b$ terimini yeniden hesaplayarak belleğin büyük kısmını, ek hesabın ise çok azını harcar.

### Gradyan biriktirme {#biriktirme}

Büyük bir batch (ör. 64) belleğe sığmıyorsa onu $J$ mikro batch'e bölüp her birinde ileri ve geri geçiş yapılır. Gradyanlar `.grad`'da **birikir** (M6.1); tüm mikro batch'ler bitince tek bir optimizasyon adımı atılır. Aynı anda yalnız bir mikro batch'in aktivasyonları bellektedir.

<figure class="fig wide"><svg viewBox="0 0 640 192" role="img" aria-label="Gradyan biriktirme: dört mikro batch sırayla ileri ve geri geçer, gradyanları ağırlıklarıyla .grad'da toplanır, sonra tek bir optimizasyon adımı atılır"><defs><marker id="g1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ink-f"/></marker></defs><rect x="20" y="30" width="100" height="30" rx="4" class="soft-a"/><text x="70" y="49" text-anchor="middle" class="sm">mikro 1: 16 örnek</text><rect x="20" y="76" width="100" height="30" rx="4" class="soft-b"/><text x="70" y="95" text-anchor="middle" class="sm">ileri + geri</text><path d="M70,60 L70,74" class="ink" marker-end="url(#g1)"/><path d="M70,106 L70,124" class="cb" marker-end="url(#g1)"/><text x="76" y="120" class="sm">× 16/64</text><rect x="140" y="30" width="100" height="30" rx="4" class="soft-a"/><text x="190" y="49" text-anchor="middle" class="sm">mikro 2: 16 örnek</text><rect x="140" y="76" width="100" height="30" rx="4" class="soft-b"/><text x="190" y="95" text-anchor="middle" class="sm">ileri + geri</text><path d="M190,60 L190,74" class="ink" marker-end="url(#g1)"/><path d="M190,106 L190,124" class="cb" marker-end="url(#g1)"/><text x="196" y="120" class="sm">× 16/64</text><rect x="260" y="30" width="100" height="30" rx="4" class="soft-a"/><text x="310" y="49" text-anchor="middle" class="sm">mikro 3: 16 örnek</text><rect x="260" y="76" width="100" height="30" rx="4" class="soft-b"/><text x="310" y="95" text-anchor="middle" class="sm">ileri + geri</text><path d="M310,60 L310,74" class="ink" marker-end="url(#g1)"/><path d="M310,106 L310,124" class="cb" marker-end="url(#g1)"/><text x="316" y="120" class="sm">× 16/64</text><rect x="380" y="30" width="100" height="30" rx="4" class="soft-a"/><text x="430" y="49" text-anchor="middle" class="sm">mikro 4: 16 örnek</text><rect x="380" y="76" width="100" height="30" rx="4" class="soft-b"/><text x="430" y="95" text-anchor="middle" class="sm">ileri + geri</text><path d="M430,60 L430,74" class="ink" marker-end="url(#g1)"/><path d="M430,106 L430,124" class="cb" marker-end="url(#g1)"/><text x="436" y="120" class="sm">× 16/64</text><rect x="20" y="128" width="460" height="28" rx="4" class="soft-p"/><text x="250" y="146" text-anchor="middle" class="sm">.grad’da birikir: Σⱼ (nⱼ/N) ∇ℓⱼ = büyük batch’in ortalama gradyanı</text><path d="M480,142 L516,142" class="ink" marker-end="url(#g1)"/><rect x="518" y="128" width="110" height="28" rx="4" class="soft-a"/><text x="573" y="146" text-anchor="middle" class="sm">opt.step(); zero_grad</text><text x="20" y="182" class="sm">Bellekte aynı anda tek mikro batch’in aktivasyonları var: tepe 4 kat küçük, sonuç aynı (BN hariç).</text></svg><figcaption>Şekil 2 — 64 örneklik batch, 4 × 16 mikro batch ile. Her mikro kayıp kendi payıyla (nⱼ/N) ölçeklenir; zero_grad yalnız adımdan sonra çağrılır.</figcaption></figure>

:::türetim Doğru ölçek: her mikro kayıp kendi payıyla
Büyük batch'in kaybı örnek ortalamasıysa ve mikro batch $j$'de $n_j$ örnek varsa
$$\mathcal L=\frac1N\sum_{i=1}^N\ell_i=\sum_j\frac{n_j}N\underbrace{\Big(\frac1{n_j}\sum_{i\in j}\ell_i\Big)}_{\mathcal L_j},\qquad\nabla\mathcal L=\sum_j\frac{n_j}N\nabla\mathcal L_j .$$
Her mikro kayıp $n_j/N$ ile çarpılıp `backward` çağrılırsa biriken gradyan büyük batch'inkiyle **birebir** aynıdır. Mikro batch'ler eşitse bu $1/J$'dir. Son parça küçükse $1/J$ yanlıştır; $n_j/N$ doğrudur. Kayıp **token ortalaması** ise (dil modelleri) aynı argüman token sayılarıyla yapılır: ağırlık $T_j/T$. Dizi uzunlukları farklıyken örnek sayısıyla ölçeklemek, kısa dizilerin tokenlarına fazla ağırlık verir.
:::

**Birebir aynılığın istisnaları.**
- **BatchNorm**: ileri geçiş mikro batch'in istatistiklerini kullanır; bu, büyük batch'in istatistiklerinden farklıdır. Hiçbir ölçekleme farkı sıfırlayamaz. Çareler: daha büyük mikro batch, LayerNorm/GroupNorm (M3.6) ya da dağıtık eğitimde senkron BN.
- **Dropout**: maskeler farklıdır ama dağılım aynıdır; beklentide fark yoktur.
- **Kayıp normalizasyonu**: yukarıdaki token sayısı tuzağı.
- **Kayan nokta**: toplama sırası değiştiği için son bitlerde fark olabilir (deneyde 10⁻¹⁶).

Aşağıdaki deney, farklı uzunluklu dizilerden oluşan bir batch'te üç ölçeklemeyi karşılaştırıyor. Token sayısıyla ölçekleme makine hassasiyetinde aynı gradyanı verir. Örnek sayısıyla ya da eşit ölçekleme bu oyuncak problemde yüzde birkaçlık, ama **sistematik** bir sapma verir. BN açıkken hiçbir ölçekleme sıfıra inmez.

::lab l2

### Diğer araçlar {#diger}

- **Karışık hassasiyet** (bf16/fp16 autocast): aktivasyonlar 16 bit saklanır, matris çarpımları hızlanır. Ağırlık tarafı (16 bayt/parametre) aynı kalır.
- **ZeRO** (Rajbhandari ve ark. 2020): $N_d$ veri paralel kopya arasında optimizasyon durumunu (aşama 1), gradyanları (aşama 2) ve ağırlıkları (aşama 3) **paylaştırır**: parametre başına bellek $16\to16/N_d$ mertebesine iner, karşılığında iletişim artar. PyTorch'ta FSDP aynı fikirdir.
- **Boşaltma** (offload): optimizasyon durumunu ya da aktivasyonları CPU belleğine taşımak (`save_on_cpu`, M6.1).
- **8 bitlik optimizasyon yöntemleri**: Adam momentlerini blok bazlı nicemleyerek 8 bayttan 2 bayta indirmek.
- **Ölçmek**: `torch.cuda.max_memory_allocated()`, `torch.cuda.memory_summary()`, bellek anlık görüntüsü (`torch.cuda.memory._record_memory_history`). "Ayrılmış" (reserved) bellek, PyTorch'un önbelleğe aldığı ama şu an kullanılmayan bloklar dahil olduğu için "kullanılan"dan (allocated) büyüktür.

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| parametre baytı | `sum(p.numel() * p.element_size() for p in model.parameters())` |
| Adam durumu | `sum(t.numel() * t.element_size() for s in opt.state.values() for t in s.values() if torch.is_tensor(t))` |
| checkpoint | `y = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)` |
| ardışık checkpoint | `torch.utils.checkpoint.checkpoint_sequential(seq, segments, x, use_reentrant=False)` |
| biriktirme | `(loss * n_j / N).backward()` her mikro batch'te; sonra `opt.step(); opt.zero_grad()` |
| DDP'de biriktirme | ara mikro batch'lerde `with model.no_sync():` (gereksiz all-reduce'u atlar) |
| tepe bellek | `torch.cuda.reset_peak_memory_stats(); ...; torch.cuda.max_memory_allocated()` |

### Özet {#ozet}

- Eğitim belleği: ağırlık + gradyan + optimizasyon durumu (Adam fp32: 16 bayt/parametre; GPT-2 small 1.99 GB) + aktivasyonlar.
- Transformer katmanı $sbh(34+5as/h)$ bayt (16 bit); dikkat terimi $s^2$ ile büyür (GPT-2 small, b = 8, s = 1024: 8.6 GB).
- Checkpointing: segment sonlarını sakla, geri geçişte yeniden hesapla. Tepe $\approx n/k+k$, en iyi $k=\sqrt n$; bedel ≈ bir ileri geçiş. El hesabı: 16 katman, k = 4 → tepe 7 (16 yerine), yeniden hesap 12.
- `checkpoint(..., use_reentrant=False)`; RNG korunur, ama BN koşu istatistikleri iki kez güncellenir.
- Gradyan biriktirme: mikro kaybı $n_j/N$ (token kaybında $T_j/T$) ile ölçekle; BN dışında büyük batch'le birebir aynı adım.
- Parametre tarafı için karışık hassasiyet, ZeRO/FSDP, boşaltma, 8 bitlik optimizasyon.
