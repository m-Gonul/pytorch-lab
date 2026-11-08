### Matris: üç bakış açısı {#matris}

Bir $n\times k$ matris $A$, $n$ satır ve $k$ sütundan oluşan bir sayı tablosudur; $A_{ij}$ $i$. satır, $j$. sütundaki elemandır (önce satır, sonra sütun). PyTorch'ta `A.shape == (n, k)`. Aynı tabloyu üç biçimde okuyacağız:

1. **Veri tablosu:** her satır bir örnek, her sütun bir öznitelik. 64 görüntülük bir batch, her biri 784 piksel: $X\in\mathbb R^{64\times784}$. Makine öğrenmesinde "satır = örnek" kuralı neredeyse evrenseldir.
2. **Sütun vektörleri yan yana:** $A=[a_{:,1}\ a_{:,2}\ \cdots\ a_{:,k}]$, her sütun $\mathbb R^n$'de bir vektör.
3. **Bir fonksiyon:** $x\mapsto Ax$, $\mathbb R^k$'dan $\mathbb R^n$'e bir **doğrusal dönüşüm**. Bu bakış, matris çarpımının neden böyle tanımlandığını açıklayan bakıştır.

:::tanım Doğrusal dönüşüm
$f$ doğrusaldır ⇔ her $x,y$ ve skaler $c$ için $f(x+y)=f(x)+f(y)$ ve $f(cx)=c\,f(x)$. Sonlu boyutta **her** doğrusal dönüşüm bir matrisle çarpmadır ve matris, dönüşümün **taban vektörlerini nereye götürdüğü** ile tamamen belirlenir: $A$'nın $j$. sütunu $= f(e_j)$. Nedeni: $x=\sum_jx_je_j$ ve doğrusallıkla $f(x)=\sum_jx_jf(e_j)$.
:::

### Matris–vektör çarpımı: satır görünümü ve sütun görünümü {#mv}

$y=Ax$, $A$ (n × k), $x$ (k), $y$ (n). Aynı sonuç iki yoldan:

- **Satır görünümü (iç çarpımlar):** $y_i=\sum_jA_{ij}x_j=(\text{A'nın }i.\text{ satırı})\cdot x$. Her çıktı, bir satırın girdiyle "ne kadar uyuştuğu"dur — $n$ nöronlu bir katmanda her nöronun ağırlık vektörü bir satırdır (M0.1).
- **Sütun görünümü (doğrusal birleşim):** $y=x_1a_{:,1}+x_2a_{:,2}+\dots+x_ka_{:,k}$. Çıktı, sütunların girdi bileşenleriyle ağırlıklandırılmış toplamıdır. Bu yüzden $Ax$'in alabileceği tüm değerler **sütun uzayı**dır.

<figure class="fig wide"><svg viewBox="0 0 560 190" role="img" aria-label="Satır görünümü ve sütun görünümü: aynı Ax çarpımının iki okunuşu"><text x="10" y="20" class="sm">Satır görünümü: y₂ = (2. satır) · x</text><g transform="translate(20,35)"><rect x="0" y="0" width="120" height="120" class="mute"/><rect x="0" y="40" width="120" height="40" class="soft-a"/><g class="mute"><line x1="40" y1="0" x2="40" y2="120"/><line x1="80" y1="0" x2="80" y2="120"/><line x1="0" y1="40" x2="120" y2="40"/><line x1="0" y1="80" x2="120" y2="80"/></g><text x="60" y="140" text-anchor="middle" class="sm">A (3 × 3)</text><rect x="140" y="0" width="40" height="120" class="soft-p"/><rect x="140" y="0" width="40" height="120" class="mute"/><text x="160" y="140" text-anchor="middle" class="sm">x</text><text x="200" y="66">=</text><rect x="220" y="0" width="40" height="120" class="mute"/><rect x="220" y="40" width="40" height="40" class="soft-a"/><text x="240" y="66" text-anchor="middle" class="ta">y₂</text><text x="240" y="140" text-anchor="middle" class="sm">y</text></g><text x="300" y="20" class="sm">Sütun görünümü: y = x₁·sütun₁ + x₂·sütun₂ + x₃·sütun₃</text><g transform="translate(300,35)"><rect x="0" y="0" width="40" height="120" class="soft-a"/><rect x="0" y="0" width="40" height="120" class="mute"/><text x="20" y="140" text-anchor="middle" class="sm">x₁ ×</text><text x="52" y="66">+</text><rect x="65" y="0" width="40" height="120" class="soft-b"/><rect x="65" y="0" width="40" height="120" class="mute"/><text x="85" y="140" text-anchor="middle" class="sm">x₂ ×</text><text x="117" y="66">+</text><rect x="130" y="0" width="40" height="120" class="soft-p"/><rect x="130" y="0" width="40" height="120" class="mute"/><text x="150" y="140" text-anchor="middle" class="sm">x₃ ×</text><text x="185" y="66">=</text><rect x="205" y="0" width="40" height="120" class="mute"/><text x="225" y="66" text-anchor="middle">y</text></g></svg><figcaption>Şekil 1 — Sol: her çıktı elemanı bir satır ile x'in iç çarpımıdır. Sağ: çıktının tamamı sütunların ağırlıklı toplamıdır. İkisi aynı sayıları verir; hangisinin düşünmeyi kolaylaştırdığı soruya bağlıdır.</figcaption></figure>

:::örnek Sayılarla
$A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$, $x=(5,6)$. Satır: $y=(1\cdot5+2\cdot6,\ 3\cdot5+4\cdot6)=(17,39)$. Sütun: $5\,(1,3)+6\,(2,4)=(5,15)+(12,24)=(17,39)$ ✓.
:::

Aşağıdaki şekilde matrisin iki sütununu (mavi ve turuncu uçlar = $Ae_1$ ve $Ae_2$) sürükle: bütün ızgara onlarla birlikte dönüşür. Yeşil $x$'i sürükle: $Ax$'e giden kesikli yol sütun görünümüdür ($x_1$ kez birinci sütun, sonra $x_2$ kez ikinci sütun).

::lab l1

**Dikkat edilecekler:** doğrusal dönüşüm orijini sabit tutar ($A0=0$), doğruları doğruya, paralel doğruları paralel doğrulara götürür, ızgara karelerini **eşit** paralelkenarlara çevirir. Birim karenin görüntüsünün alanı $|\det A|$'dır; $\det A=0$ ise düzlem bir doğruya çöker ve dönüşüm geri alınamaz (tekil matris). $\det A<0$ ise ayna görüntüsü (yönelim terslenir).

### Matris çarpımının tanımı ve shape kuralı {#tanim}

:::tanım Matris çarpımı
$A$ (n × k), $B$ (k × m) için $C=AB$ (n × m):
$$C_{ij}=\sum_{t=1}^{k}A_{it}B_{tj}=(\text{A'nın }i.\text{ satırı})\cdot(\text{B'nin }j.\text{ sütunu}).$$
**Shape kuralı:** iç boyutlar eşit olmalı ($k=k$), dış boyutlar sonucu verir: $(n\times\underline k)(\underline k\times m)\to(n\times m)$. İç boyut uyuşmazsa çarpım tanımsızdır (PyTorch: `mat1 and mat2 shapes cannot be multiplied`).
:::

:::örnek 2 × 2 elle
$A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$, $B=\begin{bmatrix}5&6\\7&8\end{bmatrix}$. $C_{11}=1\cdot5+2\cdot7=19$, $C_{12}=1\cdot6+2\cdot8=22$, $C_{21}=3\cdot5+4\cdot7=43$, $C_{22}=3\cdot6+4\cdot8=50$. Ters sırada $BA=\begin{bmatrix}23&34\\31&46\end{bmatrix}\ne AB$.
:::

**Neden böyle tanımlanmış?** Çünkü **bileşkeyi** temsil etsin diye: önce $B$ sonra $A$ uygulamak, $x\mapsto A(Bx)$, tek bir matrisle çarpmaya eşit olmalı. $(A(Bx))_i=\sum_tA_{it}(Bx)_t=\sum_tA_{it}\sum_jB_{tj}x_j=\sum_j\big(\sum_tA_{it}B_{tj}\big)x_j$ — parantez içi tam olarak $(AB)_{ij}$. Matris çarpımı = dönüşümlerin bileşkesi; sağdaki matris **önce** uygulanır.

### Dört görünüm {#gorunumler}

Aynı $C=AB$'yi dört biçimde okuyabilirsin; her biri başka bir problemde işe yarar:

| görünüm | formül | ne zaman yararlı |
|---|---|---|
| iç çarpımlar | $C_{ij}=a_{i,:}\cdot b_{:,j}$ | hesap, attention skorları $QK^\top$ |
| sütunlar | $c_{:,j}=A\,b_{:,j}$ (C'nin her sütunu A'nın sütunlarının birleşimi) | sütun uzayı, rank |
| satırlar | $c_{i,:}=a_{i,:}B$ (C'nin her satırı B'nin satırlarının birleşimi) | batch: her örnek (satır) aynı dönüşümden geçer |
| dış çarpımlar | $C=\sum_{t}a_{:,t}\,b_{t,:}$ ($k$ tane rank-1 matrisin toplamı) | SVD, LoRA ($\Delta W=BA$), gradyan $\nabla W=\sum_ix_i\delta_i^\top$ |

Dış çarpım görünümü en az bilineni ama en güçlüsüdür: $a b^\top$ (n × 1 ile 1 × m) her elemanı $a_ib_j$ olan, **rank'ı 1** olan bir matristir. $k$ küçükse $AB$ en fazla rank $k$'dır: büyük bir ağırlık değişimini iki ince matrisle ifade etmek (LoRA) bu gözleme dayanır.

### Bileşke ve sıra: neden AB ≠ BA? {#sira}

Önce 90° döndür ($R$), sonra yatayda 2 kat ger ($S$): $SR$. Ters sırada: $RS$. İkisi farklı şekil üretir:

$R=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$, $S=\begin{bmatrix}2&0\\0&1\end{bmatrix}$: $SR=\begin{bmatrix}0&-2\\1&0\end{bmatrix}$, $RS=\begin{bmatrix}0&-1\\2&0\end{bmatrix}$. $e_1=(1,0)$: $SR\,e_1=(0,1)$ (önce döndü yukarıya, gerdirme yatay olduğu için etkisiz), $RS\,e_1=(0,2)$ (önce gerildi, sonra döndü). Deney alanında "Döndürme" ve "Germe" senaryolarını sırayla dene.

**Determinant çarpımsaldır:** $\det(AB)=\det A\cdot\det B$ (alan ölçekleri çarpılır) — sıra farklı olsa da determinant aynı.

### Cebirsel özellikler ve kanıtları {#ozellikler}

1. **Birleşme:** $(AB)C=A(BC)$. *Kanıt:* $((AB)C)_{ij}=\sum_s\big(\sum_tA_{it}B_{ts}\big)C_{sj}=\sum_t\sum_sA_{it}B_{ts}C_{sj}=\sum_tA_{it}\big(\sum_sB_{ts}C_{sj}\big)=(A(BC))_{ij}$ — sonlu toplamların sırası değiştirilebilir. Sonuç aynı, **maliyet farklı** olabilir (aşağıda).
2. **Dağılma:** $A(B+C)=AB+AC$, $(A+B)C=AC+BC$ — toplamın doğrusallığı.
3. **Değişme yok:** genelde $AB\ne BA$ (yukarıdaki örnekler); shape'ler uymayabilir bile ((2×3)(3×4) var, (3×4)(2×3) yok).
4. **Birim matris:** $I$ köşegeni 1, gerisi 0: $AI=IA=A$.
5. **Transpoz kuralı:** $(AB)^\top=B^\top A^\top$ (sıra döner). *Kanıt:* $((AB)^\top)_{ij}=(AB)_{ji}=\sum_tA_{jt}B_{ti}=\sum_t(B^\top)_{it}(A^\top)_{tj}=(B^\top A^\top)_{ij}$. Shape'ler de bunu zorlar: $(n\times k)(k\times m)$'nin transpozu $(m\times n)$, ancak $(m\times k)(k\times n)$ sırasıyla elde edilir.
6. **Ters matris:** kare $A$ için $AA^{-1}=A^{-1}A=I$ — varsa (ancak $\det A\ne0$ iken). 2 × 2: $\begin{bmatrix}a&b\\c&d\end{bmatrix}^{-1}=\frac1{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$. Uygulamada tersi açıkça hesaplanmaz; $Ax=b$ doğrudan çözülür (`torch.linalg.solve`) — daha hızlı ve sayısal olarak daha kararlı.

### Broadcasting (yayın) {#broadcasting}

Shape'leri farklı iki dizi arasında **eleman eleman** işlem: küçük olan, eksik boyutlarında kopyalanmış **gibi** davranır; bellekte kopya yapılmaz (stride 0, M0.5).

:::tanım Yayın kuralları (NumPy ve PyTorch aynı)
1. Shape'ler **sağdan** hizalanır; kısa olanın soluna boyu 1 olan eksenler eklenir.
2. Her eksende boylar ya eşittir ya biri 1'dir; 1 olan diğerinin boyuna "uzatılır".
3. Aksi hâlde hata: `operands could not be broadcast together`.
:::

| A | B | sonuç | yorum |
|---|---|---|---|
| (64, 10) | (10,) | (64, 10) | her satıra aynı bias |
| (3, 1) | (1, 4) | (3, 4) | dış toplam: $C_{ij}=a_i+b_j$ |
| (8, 1, 5) | (7, 1) | (8, 7, 5) | sağdan hizala: (8,1,5) ↔ (1,7,1) |
| (64, 10) | (64,) | hata | sağdan: 10 ↔ 64 uyuşmaz |

<figure class="fig"><svg viewBox="0 0 420 170" role="img" aria-label="(3,1) ile (1,4) yayınlanarak (3,4) sonuç"><g transform="translate(10,30)"><rect x="0" y="0" width="30" height="90" class="soft-a"/><rect x="0" y="0" width="30" height="90" class="mute"/><text x="15" y="20" text-anchor="middle">1</text><text x="15" y="50" text-anchor="middle">2</text><text x="15" y="80" text-anchor="middle">3</text><text x="15" y="110" text-anchor="middle" class="sm">(3, 1)</text><text x="50" y="50">+</text><rect x="70" y="30" width="120" height="30" class="soft-b"/><rect x="70" y="30" width="120" height="30" class="mute"/><text x="85" y="50" text-anchor="middle">10</text><text x="115" y="50" text-anchor="middle">20</text><text x="145" y="50" text-anchor="middle">30</text><text x="175" y="50" text-anchor="middle">40</text><text x="130" y="80" text-anchor="middle" class="sm">(1, 4)</text><text x="205" y="50">=</text><rect x="225" y="0" width="120" height="90" class="mute"/><g class="sm"><text x="240" y="20" text-anchor="middle">11</text><text x="270" y="20" text-anchor="middle">21</text><text x="300" y="20" text-anchor="middle">31</text><text x="330" y="20" text-anchor="middle">41</text><text x="240" y="50" text-anchor="middle">12</text><text x="270" y="50" text-anchor="middle">22</text><text x="300" y="50" text-anchor="middle">32</text><text x="330" y="50" text-anchor="middle">42</text><text x="240" y="80" text-anchor="middle">13</text><text x="270" y="80" text-anchor="middle">23</text><text x="300" y="80" text-anchor="middle">33</text><text x="330" y="80" text-anchor="middle">43</text></g><text x="285" y="110" text-anchor="middle" class="sm">(3, 4)</text></g></svg><figcaption>Şekil 2 — Sütun vektörü sağa, satır vektörü aşağı "uzatılır"; her eleman a_i + b_j. Uzatma bellekte yapılmaz.</figcaption></figure>

:::yanılgı "(n,) ile (n, 1) aynı şeydir"
Değil. `a.shape == (5,)` ve `b.shape == (5, 1)` ise `a + b` (5, 5) bir matristir (dış toplam) — hata vermez, sessizce yanlış sonuç üretir. Kayıp hesabında `pred` (B, 1) ve `y` (B,) ise `(pred - y)**2` (B, B) olur ve ortalaması anlamsızdır; PyTorch `mse_loss` bu yüzden uyarı verir. Alışkanlık: işlemden önce shape'leri `assert` ile denetle.
:::

### Sinir ağında matris çarpımı {#sinir-agi}

**Doğrusal katman:** PyTorch'ta `nn.Linear(in, out)` ağırlığı (out, in) shape'inde saklar ve batch için $Y=XW^\top+b$ hesaplar: $X$ (B, in), $W^\top$ (in, out), $Y$ (B, out). Satır görünümü: her örnek aynı dönüşümden geçer; iç çarpım görünümü: $Y_{ij}=x_i\cdot w_j$, $i$. örneğin $j$. nöronla uyumu.

**Neden (out, in)?** $W$'nin her **satırı** bir nöronun ağırlık vektörüdür ve tek örnek için $y=Wx$ matematik kitaplarındaki biçimdir. Batch satır olarak geldiği için iki yanın transpozu alınır: $(Wx)^\top=x^\top W^\top$.

**Attention:** $S=QK^\top/\sqrt d$ (n × n): tüm sorgu–anahtar çiftlerinin iç çarpımları; çıktı $\mathrm{softmax}(S)V$ — ikinci bir matris çarpımı (M4.6). **Konvolüsyon:** pencereler matrisin satırlarına açılırsa (im2col) tek bir büyük matris çarpımına döner (M4.1).

### Geri yayılımda matris çarpımı {#geri}

$C=AB$, kayıp $L$, ve $G=\partial L/\partial C$ (C ile aynı shape) bilinsin. $A$ ve $B$'nin gradyanları:

:::türetim dA = G Bᵀ ve dB = Aᵀ G
$C_{ij}=\sum_tA_{it}B_{tj}$. $A_{pq}$ yalnız $p$. satırdaki $C_{pj}$'leri etkiler ve $\partial C_{pj}/\partial A_{pq}=B_{qj}$. Zincir kuralı (çok kullanılan değişkenin gradyanları toplanır):
$$\frac{\partial L}{\partial A_{pq}}=\sum_jG_{pj}B_{qj}=\sum_jG_{pj}(B^\top)_{jq}=(GB^\top)_{pq}.$$
Benzer biçimde $\frac{\partial L}{\partial B_{pq}}=\sum_iA_{ip}G_{iq}=(A^\top G)_{pq}$.

**Shape ile kontrol:** $dA$, $A$ ile aynı shape (n × k) olmalı; $G$ (n × m) ve $B^\top$ (m × k) bunu yalnız $GB^\top$ sırasıyla verir. "Tutan tek çarpım" sezgisi çoğu zaman doğru formülü verir — ama eleman eleman çarpanları (ReLU maskesi) unutmak shape'i bozmadığı için sessiz bir hatadır (M3.2 P4).
:::

Modülün **d** aşamasında $L=\sum(AB)$ için ($G$ hep 1) bu formülleri autograd ile doğrulayacaksın.

### Hesap maliyeti, bellek ve donanım {#maliyet}

**FLOP:** $C$'nin her elemanı $k$ çarpma + $k-1$ toplama: toplam ≈ $2nkm$ kayan nokta işlemi. 4096 × 4096'lık iki matris: $2\cdot4096^3\approx1.4\cdot10^{11}$ FLOP.

**Zincir sırası:** birleşme özelliği sonucu değil maliyeti değiştirir. $A$ (10 × 100), $B$ (100 × 5), $C$ (5 × 50): $(AB)C$ 7 500, $A(BC)$ 75 000 çarpma (M0.2 pratik P1). Geri yayılımda gradyan vektörünü sağdan sola çarpmak (önce matrisleri çarpmamak) tam bu yüzden ucuzdur.

**Bellek erişimi:** modern işlemcide bir çarpma bir nanosaniyenin çok altındadır; ana bellekten bir sayı getirmek ~100 kat yavaştır. Bu yüzden aynı işlemin **döngü sırası** bile hızı birkaç kat değiştirir: iç döngü belleği ardışık okursa önbellek satırları tam kullanılır. Aşağıda dört döngü sırasını kendi tarayıcında ölç.

::lab l2

**BLAS ve GPU:** NumPy/PyTorch çarpımı optimize kütüphanelere (OpenBLAS, MKL, cuBLAS) devreder. Bunlar matrisi önbelleğe sığan **bloklara** böler (her bloğu birçok kez yeniden kullanarak bellek trafiğini azaltır), SIMD komutları ve tüm çekirdekleri kullanır; GPU'da tensor çekirdekleri 16 × 16 bloklarda düşük hassasiyetli (fp16/bf16) çarpımı tek komutta yapar. **Aritmetik yoğunluk** (FLOP / taşınan bayt) büyük matris çarpımında yüksektir ($\approx n/3$): bu yüzden GPU'nun en verimli olduğu işlemdir; eleman eleman işlemler ise bellek sınırlıdır (M7).

:::ayrıntı Daha hızlı algoritmalar var mı?
Strassen (1969) 2 × 2 blok çarpımını 8 yerine 7 çarpımla yaparak $O(n^{\log_27})=O(n^{2.81})$'e iner; kuramsal en iyi üs bugün ~2.37'dir. Pratikte Strassen yalnız çok büyük matrislerde ve sayısal kararlılığı biraz feda ederek kullanılır; derin öğrenme kütüphaneleri klasik $O(n^3)$ algoritmayı donanıma göre bloklayarak kullanır — sabitler ve bellek erişimi asimptotik üsten daha önemlidir.
:::

### Batch matris çarpımı ve einsum dili {#einsum}

Birçok bağımsız çarpım tek çağrıda: `A @ B`, $A$ (B, n, k) ve $B$ (B, k, m) için (B, n, m) — en sağdaki iki eksen matris, soldakiler batch (yayın kurallarına uyar). Attention'da (batch, baş, n, d) tensörleri böyle çarpılır.

**einsum:** her eksene bir harf ver, çıktıda geçmeyen harfler toplanır:

| işlem | einsum |
|---|---|
| iç çarpım | `"i,i->"` |
| matris–vektör | `"ij,j->i"` |
| matris çarpımı | `"ik,kj->ij"` |
| transpoz | `"ij->ji"` |
| batch matmul | `"bik,bkj->bij"` |
| attention skorları | `"bhnd,bhmd->bhnm"` |
| dış çarpım | `"i,j->ij"` |
| köşegen / iz | `"ii->i"` / `"ii->"` |

Okuması kolay, hata yapması zor (shape'ler dizgide yazılı) ve kütüphane en iyi sırayı seçebilir (`opt_einsum`).

### Sayısal konular {#sayisal}

- **Birikim hassasiyeti:** $k$ terimlik toplamda yuvarlama hatası ~$\sqrt k\,\epsilon$ büyür. GPU'larda fp16/bf16 çarpımlar **fp32 biriktiriciyle** toplanır; yine de büyük $k$'da sonuçlar CPU'dan son basamaklarda farklıdır.
- **TF32:** Ampere ve sonrası GPU'larda PyTorch fp32 matmul'ları varsayılan olarak 10 bitlik mantisle (TF32) yapabilir (`torch.backends.cuda.matmul.allow_tf32`): hız için hassasiyet. Testlerde toleransı buna göre seç.
- **Deterministiklik:** paralel indirgeme sırası koşudan koşuya değişebilir; bit düzeyinde aynı sonuç gerekiyorsa `torch.use_deterministic_algorithms(True)`.

### Özet ve sembol tablosu {#ozet}

| sembol | anlamı |
|---|---|
| $A\in\mathbb R^{n\times k}$ | n satır, k sütun |
| $A_{ij}$ | i. satır, j. sütun |
| $a_{i,:}$, $a_{:,j}$ | i. satır, j. sütun vektörü |
| $AB$ | bileşke: önce $B$, sonra $A$ |
| $A^\top$ | transpoz, $(A^\top)_{ij}=A_{ji}$ |
| $I$ | birim matris |
| $\det A$ | alan/hacim ölçeği (işaret: yönelim) |

**Akılda kalsın:** matris çarpımı = iç çarpımların tablosu = dönüşümlerin bileşkesi. Shape kuralı iç boyutların eşleşmesidir; geri yayılımın formülleri ($GB^\top$, $A^\top G$) shape'lerden okunabilir; aynı FLOP'u yapan farklı sıralamalar bellek erişimi yüzünden çok farklı hızda çalışır.

**Kaynaklar:** Strang, *Introduction to Linear Algebra* Bölüm 2.4 (matris işlemleri kuralları) ve MIT 18.06 Ders 3; 3Blue1Brown, *Essence of Linear Algebra* 3–4 (doğrusal dönüşümler, bileşke olarak çarpım); Goodfellow ve ark., *Deep Learning* Bölüm 2.1–2.3; d2l.ai §2.3; NumPy *Broadcasting* belgesi; Goto & van de Geijn (2008), "Anatomy of high-performance matrix multiplication".
