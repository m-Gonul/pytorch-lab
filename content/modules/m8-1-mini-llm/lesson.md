### Proje: toplamayı öğrenen küçük bir GPT {#proje}

Bu modül önceki fazların araçlarını tek bir eğitim koşusunda birleştiriyor: M4.7'deki GPT, M3.4'teki optimizasyon, M6'daki profil çıkarma ve M6.5/M7'deki hızlandırma. Görev küçük ama gerçekçidir: iki basamaklı iki sayının toplamını yalnız örneklerden öğrenen bir dil modeli (Karpathy'nin minGPT "adder" örneğine dayanır). Model aritmetiği bilmez; yalnız karakter dizilerinde bir sonraki karakteri tahmin etmeyi öğrenir. Bir eğitim koşusunun her adımı burada görünür: bütçe, veri biçimi, kayıp maskesi, öğrenme oranı programı, değerlendirme ve profil.

### Bütçe: parametre, token, FLOP {#butce}

Bir koşuyu başlatmadan önce maliyeti tahmin etmek gerekir.

:::türetim Eğitim FLOP'u ≈ 6ND
Bir token için ileri geçişte her ağırlık bir çarpma ve bir toplamaya girer: ≈ $2N$ FLOP (dikkat skorlarının $2LTd$ terimi küçük bağlamlarda ihmal edilir). Geri geçiş, her ağırlık için iki VJP (girdiye ve ağırlığa göre) gerektirdiği için ≈ $4N$ FLOP'tur (M6.1). Token başına toplam ≈ $6N$, $D$ token için
$$C\approx6ND .$$
Süre $=C/(\text{tepe}\cdot\text{MFU})$. MFU (model FLOP kullanımı), donanım tepesinin modelin gerçekten kullandığı oranıdır.
:::

**Ne kadar veri?** Hoffmann ve ark.'nın (2022) Chinchilla çalışması, sabit bir hesap bütçesinde kaybı en küçükleyen model boyu ile token sayısının **birlikte** ölçeklenmesi gerektiğini gösterdi; kabaca $D\approx20N$. (Daha önceki Kaplan ve ark. 2020 önerisi modelleri büyük ve az eğitilmiş bırakıyordu.)

**El hesabı.** $N=10^7$: Chinchilla-optimal $D\approx20N=\mathbf{2\cdot10^8}$ token; $C=6ND=\mathbf{1.2\cdot10^{16}}$ FLOP; 100 TFLOP/s tepe ve %40 MFU ile $1.2\cdot10^{16}/(4\cdot10^{13})=\mathbf{300}$ s.

Aşağıdaki deney, sabit bir $C$ için kayıp–model boyu eğrisini (eş-FLOP eğrisi) Chinchilla'nın uydurduğu kayıp yasasıyla çiziyor: $L(N,D)=E+A/N^\alpha+B/D^\beta$. Eğrinin dibi en iyi bölüşümdür ve dipte düz olduğu için "20 token/parametre" kuralından küçük sapmalar az kayıp getirir.

::lab l1

### Görev biçimi {#bicim}

Her satır sabit 10 karakterdir: `ab+cd=` sonra cevabın üç basamağı **ters sırada** ve satır sonu. $07+45=52$ için cevap `052`, ters `250`: satır `07+45=250⏎`. Sözlük 13 token'dır: 0–9, `+`, `=`, satır sonu.

<figure class="fig wide"><svg viewBox="0 0 640 202" role="img" aria-label="Veri biçimi: 07+45=250 ve satır sonu; girdi x ilk dokuz token, hedef y son dokuz token; kayıp yalnız cevap basamaklarının konumlarında"><text x="10" y="18" class="sm">Satır: 07 + 45 = 52, cevap üç basamak ve ters: 052 → “250”. Token’lar: 0–9, + (10), = (11), satır sonu (12).</text><text x="82" y="40" text-anchor="middle" class="sm">0</text><rect x="60" y="46" width="44" height="28" rx="3" class="soft-a" stroke="var(--line)"/><text x="82" y="65" text-anchor="middle" class="ink">0</text><text x="134" y="40" text-anchor="middle" class="sm">1</text><rect x="112" y="46" width="44" height="28" rx="3" class="soft-a" stroke="var(--line)"/><text x="134" y="65" text-anchor="middle" class="ink">7</text><text x="186" y="40" text-anchor="middle" class="sm">2</text><rect x="164" y="46" width="44" height="28" rx="3" class="soft-a" stroke="var(--line)"/><text x="186" y="65" text-anchor="middle" class="ink">+</text><text x="238" y="40" text-anchor="middle" class="sm">3</text><rect x="216" y="46" width="44" height="28" rx="3" class="soft-a" stroke="var(--line)"/><text x="238" y="65" text-anchor="middle" class="ink">4</text><text x="290" y="40" text-anchor="middle" class="sm">4</text><rect x="268" y="46" width="44" height="28" rx="3" class="soft-a" stroke="var(--line)"/><text x="290" y="65" text-anchor="middle" class="ink">5</text><text x="342" y="40" text-anchor="middle" class="sm">5</text><rect x="320" y="46" width="44" height="28" rx="3" class="soft-a" stroke="var(--line)"/><text x="342" y="65" text-anchor="middle" class="ink">=</text><text x="394" y="40" text-anchor="middle" class="sm">6</text><rect x="372" y="46" width="44" height="28" rx="3" class="soft-b" stroke="var(--line)"/><text x="394" y="65" text-anchor="middle" class="ink">2</text><text x="446" y="40" text-anchor="middle" class="sm">7</text><rect x="424" y="46" width="44" height="28" rx="3" class="soft-b" stroke="var(--line)"/><text x="446" y="65" text-anchor="middle" class="ink">5</text><text x="498" y="40" text-anchor="middle" class="sm">8</text><rect x="476" y="46" width="44" height="28" rx="3" class="soft-b" stroke="var(--line)"/><text x="498" y="65" text-anchor="middle" class="ink">0</text><text x="550" y="40" text-anchor="middle" class="sm">9</text><rect x="528" y="46" width="44" height="28" rx="3" class="soft-a" stroke="var(--line)"/><text x="550" y="65" text-anchor="middle" class="ink">⏎</text><text x="52" y="65" text-anchor="end" class="sm">satır</text><rect x="60" y="92" width="44" height="24" rx="3" class="soft-a" stroke="var(--line)"/><text x="82" y="109" text-anchor="middle" class="sm">0</text><rect x="60" y="124" width="44" height="24" rx="3" class="mute-f" stroke="var(--line)" fill-opacity="0.3"/><text x="82" y="141" text-anchor="middle" class="sm">7</text><rect x="112" y="92" width="44" height="24" rx="3" class="soft-a" stroke="var(--line)"/><text x="134" y="109" text-anchor="middle" class="sm">7</text><rect x="112" y="124" width="44" height="24" rx="3" class="mute-f" stroke="var(--line)" fill-opacity="0.3"/><text x="134" y="141" text-anchor="middle" class="sm">+</text><rect x="164" y="92" width="44" height="24" rx="3" class="soft-a" stroke="var(--line)"/><text x="186" y="109" text-anchor="middle" class="sm">+</text><rect x="164" y="124" width="44" height="24" rx="3" class="mute-f" stroke="var(--line)" fill-opacity="0.3"/><text x="186" y="141" text-anchor="middle" class="sm">4</text><rect x="216" y="92" width="44" height="24" rx="3" class="soft-a" stroke="var(--line)"/><text x="238" y="109" text-anchor="middle" class="sm">4</text><rect x="216" y="124" width="44" height="24" rx="3" class="mute-f" stroke="var(--line)" fill-opacity="0.3"/><text x="238" y="141" text-anchor="middle" class="sm">5</text><rect x="268" y="92" width="44" height="24" rx="3" class="soft-a" stroke="var(--line)"/><text x="290" y="109" text-anchor="middle" class="sm">5</text><rect x="268" y="124" width="44" height="24" rx="3" class="mute-f" stroke="var(--line)" fill-opacity="0.3"/><text x="290" y="141" text-anchor="middle" class="sm">=</text><rect x="320" y="92" width="44" height="24" rx="3" class="soft-a" stroke="var(--line)"/><text x="342" y="109" text-anchor="middle" class="sm">=</text><rect x="320" y="124" width="44" height="24" rx="3" class="soft-p" stroke="var(--line)"/><text x="342" y="141" text-anchor="middle" class="sm">2</text><rect x="372" y="92" width="44" height="24" rx="3" class="soft-a" stroke="var(--line)"/><text x="394" y="109" text-anchor="middle" class="sm">2</text><rect x="372" y="124" width="44" height="24" rx="3" class="soft-p" stroke="var(--line)"/><text x="394" y="141" text-anchor="middle" class="sm">5</text><rect x="424" y="92" width="44" height="24" rx="3" class="soft-a" stroke="var(--line)"/><text x="446" y="109" text-anchor="middle" class="sm">5</text><rect x="424" y="124" width="44" height="24" rx="3" class="soft-p" stroke="var(--line)"/><text x="446" y="141" text-anchor="middle" class="sm">0</text><rect x="476" y="92" width="44" height="24" rx="3" class="soft-a" stroke="var(--line)"/><text x="498" y="109" text-anchor="middle" class="sm">0</text><rect x="476" y="124" width="44" height="24" rx="3" class="mute-f" stroke="var(--line)" fill-opacity="0.3"/><text x="498" y="141" text-anchor="middle" class="sm">⏎</text><text x="52" y="109" text-anchor="end" class="sm">x</text><text x="52" y="141" text-anchor="end" class="sm">y</text><text x="10" y="174" class="sm">Kayıp yalnız y[:, 5:8]’de (yeşil): model “=”den sonra cevap basamaklarını tahmin eder; soruyu tahmin etmesi istenmez.</text><text x="10" y="192" class="sm">Ters sıra: önce birler basamağı üretilir; her basamak, hizalı iki basamak ve bir önceki adımın eldesiyle belirlenir.</text></svg><figcaption>Şekil 1 — Görev biçimi ve kayıp maskesi. El hesabı: 38 + 67 = 105, ters cevap “501”. Batch 128'de adım başına 128 × 9 = 1152 girdi token'ı.</figcaption></figure>

**El hesabı.** $38+67=105$ → ters `501`. Batch 128'de adım başına $128\cdot9=\mathbf{1152}$ girdi token'ı (satırın ilk 9 karakteri girdi, son 9 hedef).

:::sezgi Neden ters sıra?
İnsanlar toplamayı sağdan sola yapar: birler basamağı hemen bulunur, elde bir sonraki basamağa taşınır. Normal sırada model önce **en büyük** basamağı üretmek zorundadır; bu, bütün eldelerin önceden hesaplanmasını gerektirir. Ters sırada her adımda yalnız hizalı iki basamak ve bir önceki basamağın eldesi gerekir; bu bilgi ya girdide ya da az önce üretilen çıktının yanında durur. Lee ve ark. (2023) küçük transformer'ların ters sırayla çok daha az örnekle ve daha kararlı öğrendiğini gösterdi. Biçim seçimi, modelin öğrenmesi gereken fonksiyonun karmaşıklığını değiştirir.
:::

**Kayıp maskesi.** Soru kısmı (`ab+cd=`) rastgeledir; modelden onu tahmin etmesi istenmez. Kayıp yalnız cevap basamaklarının hedef konumlarında (`y[:, 5:8]`) hesaplanır. Böylece modelin kapasitesi öğrenilebilir kısma gider ve kayıp değeri doğrudan cevabın kalitesini ölçer.

**Taban.** Cevap basamaklarını soruya hiç bakmadan, yalnız marjinal dağılımlarından tahmin eden bir "model" ne kadar kayıp verir? Birler ve onlar basamakları $\{0,\dots,9\}$'da düzgündür ($\ln10=2.303$ nat), yüzler basamağı 0 ya da 1'dir ($P(a+b\ge100)=0.495$, $\approx0.693$ nat). Toplam 5.298 nat, konum başına ortalama **1.766**. Eğitim kaybı bunun altına inmiyorsa model soruyu kullanmıyor demektir. Aşama c bunu NumPy'da hesaplıyor.

### Eğitim döngüsü {#dongu}

Aşama d, bir GPT'yi şu parçalarla eğitiyor:
- **AdamW** (ağırlık bozunumu 0.01; M3.4).
- **Isınma + kosinüs**: 100 adımda doğrusal ısınma, sonra $\eta_{\max}$'tan $\eta_{\min}=\eta_{\max}/10$'a kosinüs azalma.
- **Gradyan kırpma** (norm 1): ısınma bitmeden gelen büyük gradyanlara karşı sigorta.
- **Değerlendirme**: kayıp değil, **tam eşleşme doğruluğu**. Model `ab+cd=` ile başlatılır ve 3 adım greedy üretir (batch üzerinde vektörel); üç basamağın üçü de doğruysa doğru sayılır. %95+ hedeflenir.

<figure class="fig wide"><svg viewBox="0 0 640 210" role="img" aria-label="Öğrenme oranı programı: 100 adım doğrusal ısınma, sonra 10 üzeri eksi 3'ten 10 üzeri eksi 4'e kosinüs azalma; adım 49'da 5 çarpı 10 üzeri eksi 4, adım 600'de 5.5 çarpı 10 üzeri eksi 4"><path d="M60,170 L600,170" class="mute"/><path d="M60,16 L60,170" class="mute"/><path d="M60.0,168.6 L62.5,161.6 L64.9,154.6 L67.4,147.6 L69.8,140.6 L72.3,133.6 L74.7,126.6 L77.2,119.6 L79.6,112.6 L82.1,105.6 L84.5,98.6 L87.0,91.6 L89.5,84.6 L91.9,77.6 L94.4,70.6 L96.8,63.6 L99.3,56.6 L101.7,49.6 L104.2,42.6 L106.6,35.6 L109.1,30.0 L111.5,30.0 L114.0,30.0 L116.5,30.1 L118.9,30.1 L121.4,30.2 L123.8,30.3 L126.3,30.4 L128.7,30.5 L131.2,30.6 L133.6,30.8 L136.1,30.9 L138.5,31.1 L141.0,31.3 L143.5,31.5 L145.9,31.7 L148.4,32.0 L150.8,32.2 L153.3,32.5 L155.7,32.8 L158.2,33.1 L160.6,33.4 L163.1,33.7 L165.5,34.1 L168.0,34.4 L170.5,34.8 L172.9,35.2 L175.4,35.6 L177.8,36.0 L180.3,36.4 L182.7,36.9 L185.2,37.3 L187.6,37.8 L190.1,38.3 L192.5,38.8 L195.0,39.3 L197.5,39.8 L199.9,40.3 L202.4,40.9 L204.8,41.5 L207.3,42.0 L209.7,42.6 L212.2,43.2 L214.6,43.8 L217.1,44.5 L219.5,45.1 L222.0,45.7 L224.5,46.4 L226.9,47.1 L229.4,47.8 L231.8,48.5 L234.3,49.2 L236.7,49.9 L239.2,50.6 L241.6,51.3 L244.1,52.1 L246.5,52.8 L249.0,53.6 L251.5,54.4 L253.9,55.2 L256.4,56.0 L258.8,56.8 L261.3,57.6 L263.7,58.4 L266.2,59.2 L268.6,60.1 L271.1,60.9 L273.5,61.8 L276.0,62.6 L278.5,63.5 L280.9,64.4 L283.4,65.3 L285.8,66.2 L288.3,67.1 L290.7,68.0 L293.2,68.9 L295.6,69.8 L298.1,70.7 L300.5,71.7 L303.0,72.6 L305.5,73.5 L307.9,74.5 L310.4,75.4 L312.8,76.4 L315.3,77.3 L317.7,78.3 L320.2,79.3 L322.6,80.2 L325.1,81.2 L327.5,82.2 L330.0,83.1 L332.5,84.1 L334.9,85.1 L337.4,86.1 L339.8,87.1 L342.3,88.1 L344.7,89.0 L347.2,90.0 L349.6,91.0 L352.1,92.0 L354.5,93.0 L357.0,94.0 L359.5,95.0 L361.9,96.0 L364.4,97.0 L366.8,97.9 L369.3,98.9 L371.7,99.9 L374.2,100.9 L376.6,101.9 L379.1,102.9 L381.5,103.8 L384.0,104.8 L386.5,105.8 L388.9,106.7 L391.4,107.7 L393.8,108.7 L396.3,109.6 L398.7,110.6 L401.2,111.5 L403.6,112.5 L406.1,113.4 L408.5,114.3 L411.0,115.3 L413.5,116.2 L415.9,117.1 L418.4,118.0 L420.8,118.9 L423.3,119.8 L425.7,120.7 L428.2,121.6 L430.6,122.5 L433.1,123.4 L435.5,124.2 L438.0,125.1 L440.5,125.9 L442.9,126.8 L445.4,127.6 L447.8,128.4 L450.3,129.2 L452.7,130.0 L455.2,130.8 L457.6,131.6 L460.1,132.4 L462.5,133.2 L465.0,133.9 L467.5,134.7 L469.9,135.4 L472.4,136.1 L474.8,136.8 L477.3,137.5 L479.7,138.2 L482.2,138.9 L484.6,139.6 L487.1,140.3 L489.5,140.9 L492.0,141.5 L494.5,142.2 L496.9,142.8 L499.4,143.4 L501.8,144.0 L504.3,144.5 L506.7,145.1 L509.2,145.7 L511.6,146.2 L514.1,146.7 L516.5,147.2 L519.0,147.7 L521.5,148.2 L523.9,148.7 L526.4,149.1 L528.8,149.6 L531.3,150.0 L533.7,150.4 L536.2,150.8 L538.6,151.2 L541.1,151.6 L543.5,151.9 L546.0,152.3 L548.5,152.6 L550.9,152.9 L553.4,153.2 L555.8,153.5 L558.3,153.8 L560.7,154.0 L563.2,154.3 L565.6,154.5 L568.1,154.7 L570.5,154.9 L573.0,155.1 L575.5,155.2 L577.9,155.4 L580.4,155.5 L582.8,155.6 L585.3,155.7 L587.7,155.8 L590.2,155.9 L592.6,155.9 L595.1,156.0 L597.5,156.0 L600.0,156.0" class="ca w2"/><circle cx="84.1" cy="100.0" r="4" class="cb-f"/><text x="92.1" y="94.0" class="tb">adım 49: 5·10⁻⁴</text><circle cx="354.5" cy="93.0" r="4" class="cb-f"/><text x="362.5" y="87.0" class="tb">adım 600: 5.5·10⁻⁴</text><text x="54" y="174.0" text-anchor="end" class="sm">0</text><text x="54" y="104.0" text-anchor="end" class="sm">0.0005</text><text x="54" y="34.0" text-anchor="end" class="sm">0.001</text><text x="60.0" y="186" text-anchor="middle" class="sm">0</text><text x="109.1" y="186" text-anchor="middle" class="sm">100</text><text x="354.5" y="186" text-anchor="middle" class="sm">600</text><text x="600.0" y="186" text-anchor="middle" class="sm">1100</text><text x="330.0" y="202" text-anchor="middle" class="sm">adım (ısınma 100, toplam 1100, η<tspan baseline-shift="-3" font-size="75%">max</tspan> = 10⁻³, η<tspan baseline-shift="-3" font-size="75%">min</tspan> = 10⁻⁴)</text></svg><figcaption>Şekil 2 — Isınma + kosinüs programı, formülden hesaplandı (modülün el hesapları). Isınma, Adam'ın ilk adımlardaki güvenilmez ikinci moment kestirimini ve büyük ilk gradyanları yumuşatır.</figcaption></figure>

**El hesabı.** Isınma 100, toplam 1100, $\eta_{\max}=10^{-3}$, $\eta_{\min}=10^{-4}$. Adım 49'da (ısınma) $\eta=10^{-3}\cdot50/100=\mathbf{5\cdot10^{-4}}$. Adım 600'de $p=(600-100)/1000=0.5$, $\eta=10^{-4}+\frac12\cdot9\cdot10^{-4}(1+\cos\frac\pi2)=\mathbf{5.5\cdot10^{-4}}$.

:::uyarı Kayıp düşüyor ama doğruluk düşük
Ortalama kayıp birkaç basamakta çok iyi, birinde kötü olabilir; tam eşleşme ise üç basamağın hepsini ister. Eğitim kaybı tabanın (1.766) belirgin altında ama doğruluk düşükse, genellikle model eldeyi gerektiren durumları (ör. $a_0+b_0\ge10$) henüz öğrenmemiştir. Doğruluğu elde olan ve olmayan örneklere ayırarak ölçmek hatayı yerelleştirir.
:::

### Profil çıkarmak ve hızlandırmak {#profil}

Eğitim çalışıyorsa sıradaki soru "zaman nereye gidiyor?"dur. `torch.profiler.profile` bir kod bloğunun içindeki işlemleri (ve GPU'da çekirdekleri) zamanlar; `key_averages()` işlem adına göre toplar.

```python
with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
    step()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
```

Aşama e'deki `LoopAttention`, her dikkat başlığı için ayrı `q`, `k`, `v` doğrusal katmanları kurar ve başlıklar üzerinde bir Python döngüsü yapar. Profil, zamanın çoğunun çok sayıda küçük `addmm`, `matmul` ve `softmax` çağrısına gittiğini gösterir: model küçük olduğu için süre **hesaptan değil ek yükten** gelir (M6.5'teki üç kaynak). Çözüm, başlık döngüsünü tek bir batch'li işleme indirmektir:
- Bütün başlıkların $q$, $k$, $v$ ağırlıkları yığılarak tek bir `qkv = nn.Linear(d, 3d)` kurulur (çıktı sırası: bütün $q$'lar, bütün $k$'lar, bütün $v$'ler).
- Çıktı `(B, T, 3, H, d_h)` biçimine getirilip başlık boyutu batch boyutu gibi taşınır.
- `F.scaled_dot_product_attention(q, k, v, is_causal=True)` tek çağrıda bütün başlıkları hesaplar (uygun donanımda FlashAttention çekirdeğiyle).

Sonuç sayısal olarak aynıdır (aynı ağırlıklar, aynı işlemler), çağrı sayısı başlık sayısı kadar azalır. Bu, M6.5'teki "çekirdek başlatma ve Python ek yükünü azalt" ilkesinin model düzeyindeki hâlidir. Ölçmenin iki kuralı: ilk çağrıları (ısınma) ölçüme katmamak, GPU'da `torch.cuda.synchronize()` ile beklemek.

**MFU'yu hesaplamak.** Profilden sonra koşunun ne kadar verimli olduğunu tek bir sayıyla özetlemek için MFU kullanılır. Aşağıdaki deney model yapılandırmasından $N$'yi, token başına FLOP'u ve ölçülen adım süresinden MFU'yu hesaplıyor. GPT-2 small yapılandırması $N\approx1.24\cdot10^8$ verir. Küçük modellerde MFU çok düşüktür: süre ek yük bağımlıdır.

::lab l2

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| maskeli kayıp | `F.cross_entropy(logits[:, 5:8].reshape(-1, V), y[:, 5:8].reshape(-1))` |
| program | `torch.optim.lr_scheduler.LambdaLR(opt, lambda t: lr_at(t) / max_lr)` |
| kırpma | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` |
| greedy üretim | `for _ in range(3): nxt = model(x)[:, -1].argmax(-1, keepdim=True); x = torch.cat([x, nxt], 1)` |
| profil | `with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p: ...`; `p.key_averages().table(sort_by="self_cpu_time_total")` |
| birleşik dikkat | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` |
| hızlandırma | `torch.compile(model)`; `torch.autocast("cuda", dtype=torch.bfloat16)` |

### Özet {#ozet}

- Bütçe: $C\approx6ND$ (ileri $2N$, geri $4N$); Chinchilla $D\approx20N$. El hesabı: $N=10^7$ → $2\cdot10^8$ token, $1.2\cdot10^{16}$ FLOP, 300 s.
- Görev biçimi öğrenilebilirliği belirler: ters cevap (38 + 67 → "501") her basamağı yerel yapar. Kayıp yalnız cevap konumlarında; taban 1.766 nat/konum.
- Eğitim: AdamW, ısınma + kosinüs (adım 49: 5·10⁻⁴, adım 600: 5.5·10⁻⁴), kırpma, tam eşleşme doğruluğu.
- Profil: küçük modelde süre ek yük bağımlı; başlık döngüsünü tek QKV matmul + SDPA'ya indirmek çağrı sayısını azaltır, sonucu değiştirmez. MFU ile verimi özetle.
