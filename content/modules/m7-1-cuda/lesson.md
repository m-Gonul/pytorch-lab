### GPU neden farklı düşünür? {#gpu}

Bir CPU az sayıda (8–128) güçlü çekirdekle **gecikmeyi** en aza indirmeye çalışır: büyük önbellekler, dal tahmini, sıra dışı yürütme. Bir GPU ise binlerce basit hesap birimiyle **verimi** en üst düzeye çıkarır. Tek bir işi hızlı yapmaz; aynı işi çok sayıda veri üzerinde aynı anda yapar. Bellek gecikmesini önbellekle değil, bekleyen iş parçacıkları arasında hızla geçiş yaparak gizler.

Bir NVIDIA GPU'su **akış çoklu işlemcilerinden** (SM) oluşur; A100'de 108 SM vardır. Her SM'nin kendi yazmaç dosyası, shared memory/L1 önbelleği ve hesap birimleri vardır. Bir SM'deki iş parçacıkları (thread) 32'lik **warp**'lar hâlinde zamanlanır. Bir warp'ın 32 thread'i aynı komutu aynı anda, farklı verilerle yürütür (SIMT: tek komut, çok thread). Bir warp bellek beklerken SM başka bir warp'a geçer. Yeterince çok warp hazırda bekliyorsa (yüksek **doluluk**, occupancy) gecikme gizlenir.

### Yürütme modeli: grid, block, thread {#model}

CUDA'da bir **çekirdek** (kernel), bir thread'in yapacağı işi tanımlayan bir fonksiyondur. Çağrılırken kaç thread'le çalışacağı söylenir: thread'ler **block**'lara, block'lar bir **grid**'e gruplanır (`kernel<<<grid, block>>>(...)`).
- Bir block'un thread'leri aynı SM'de çalışır, shared memory'yi paylaşır ve `__syncthreads()` ile eşzamanlanabilir. Block başına en fazla 1024 thread olabilir.
- Farklı block'lar birbirinden bağımsızdır; herhangi bir sırada, herhangi bir SM'de çalışabilir. Bu bağımsızlık, aynı programın 10 SM'li ve 132 SM'li GPU'da ölçeklenmesini sağlar.
- Her thread kendi konumunu yerleşik değişkenlerden okur: `blockIdx`, `blockDim`, `threadIdx` (1, 2 ya da 3 boyutlu). Tek boyutta global indeks $i=\texttt{blockIdx.x}\cdot\texttt{blockDim.x}+\texttt{threadIdx.x}$.

<figure class="fig wide"><svg viewBox="0 0 640 200" role="img" aria-label="N = 1000 elemanlık vektör toplama: 256 thread'lik 4 block, son blokta 24 thread boşta; blok 2'nin 5. thread'i 517. elemanı işler"><rect x="10" y="30" width="150" height="40" rx="4" class="soft-a" stroke="var(--line)"/><text x="85.0" y="22" text-anchor="middle" class="sm">blockIdx.x = 0</text><text x="85.0" y="55" text-anchor="middle" class="sm">i = 0 … 255</text><rect x="166" y="30" width="150" height="40" rx="4" class="soft-a" stroke="var(--line)"/><text x="241.0" y="22" text-anchor="middle" class="sm">blockIdx.x = 1</text><text x="241.0" y="55" text-anchor="middle" class="sm">i = 256 … 511</text><rect x="322" y="30" width="150" height="40" rx="4" class="soft-a" stroke="var(--line)"/><text x="397.0" y="22" text-anchor="middle" class="sm">blockIdx.x = 2</text><text x="397.0" y="55" text-anchor="middle" class="sm">i = 512 … 767</text><rect x="478" y="30" width="150" height="40" rx="4" class="soft-a" stroke="var(--line)"/><text x="553.0" y="22" text-anchor="middle" class="sm">blockIdx.x = 3</text><text x="553.0" y="55" text-anchor="middle" class="sm">i = 768 … 1023</text><rect x="613.9" y="30" width="14.1" height="40" class="soft-b"/><text x="628" y="88" text-anchor="end" class="tb">24 thread boşta (i ≥ 1000)</text><path d="M324.9,30 L324.9,70" class="cp w3"/><text x="328.9" y="104" class="tp">threadIdx.x = 5 → i = 2·256 + 5 = 517</text><text x="10" y="134" class="sm">__global__ void add(const float* x, const float* y, float* out, int n) {</text><text x="10" y="150" class="sm">    int i = blockIdx.x * blockDim.x + threadIdx.x;   if (i &lt; n) out[i] = x[i] + y[i]; }</text><text x="10" y="166" class="sm">add&lt;&lt;&lt;(n + 255) / 256, 256&gt;&gt;&gt;(x, y, out, n);   // ⌈1000 / 256⌉ = 4 block</text><text x="10" y="190" class="sm">Bir block bir SM’de çalışır; içindeki thread’ler 32’lik warp’lar hâlinde aynı komutu birlikte yürütür (SIMT).</text></svg><figcaption>Şekil 1 — Izgara, block ve thread; modülün el hesabı. Sınır kontrolü olmasaydı son blokun 24 thread'i dizinin dışına yazardı.</figcaption></figure>

**El hesabı.** $N=1000$, `blockDim.x = 256`: $\lceil1000/256\rceil=\mathbf4$ block, $4\cdot256=1024$ thread; $1024-1000=\mathbf{24}$ thread sınır kontrolüne (`if (i < n)`) takılır. `blockIdx.x = 2`, `threadIdx.x = 5`: $i=2\cdot256+5=\mathbf{517}$.

:::uyarı Sınır kontrolü unutulursa
Son block'un fazla thread'leri dizinin dışına yazar. Bu her zaman bir hata vermez: bitişik bellekteki başka bir tensörü sessizce bozabilir. `compute-sanitizer` bu tür hataları yakalar. Thread başına birden çok eleman işleyen "grid-stride" döngüleri (`for (i = …; i < n; i += gridDim.x * blockDim.x)`) hem sınırı hem grid boyutundan bağımsızlığı birlikte çözer.
:::

### Bellek birleştirme {#coalescing}

Global belleğe (HBM) erişim 32 baytlık **sektörler** hâlinde yapılır. Bir warp'ın 32 thread'i bir yükleme komutu çalıştırdığında donanım, thread'lerin dokunduğu sektörleri toplar ve onları getirir. Ardışık thread'ler ardışık adreslere erişiyorsa (fp32'de 32 × 4 = 128 bayt = 4 sektör) erişim **birleşiktir** (coalesced): her getirilen bayt kullanılır. Thread'ler arasında adım varsa getirilen sektörlerin çoğu kullanılmayan veri taşır.

:::türetim Adımlı erişimin verimi
Thread $t$, $(o+t\cdot s)$ numaralı elemanı okusun (eleman boyu $e$ bayt). Gereken bayt $32e$'dir. $se\ge32$ ise her thread ayrı bir sektöre düşer: 32 sektör, $32\cdot32$ bayt getirilir ve verim $e/32$ olur (fp32'de %12.5). $se<32$ iken yaklaşık $\lceil32se/32\rceil$ sektör getirilir; verim $\approx1/s$'dir. Satır-ana bir matrisin **sütununu** thread başına okumak tam bu durumdur ($s=n$). Çözüm, erişimi satır boyunca yapmak ya da tile'ı shared memory'ye birleşik okuyup orada transpoze etmektir.
:::

::lab l1

### Döşeme (tiling) ve shared memory {#tiling}

Naif matris çarpımı çekirdeğinde her thread bir $C_{ij}$ hesaplar: $A$'nın $i$. satırını ve $B$'nin $j$. sütununu global bellekten okur. Toplam okuma $n^2\cdot2n=2n^3$ eleman olur. Oysa her $A$ elemanı $n$ farklı thread tarafından kullanılır; aynı veri $n$ kez taşınır.

<figure class="fig wide"><svg viewBox="0 0 640 196" role="img" aria-label="GPU bellek hiyerarşisi: yazmaçlar, shared memory, L2 ve HBM; tiled matmulda her block bir C tile'ı hesaplar ve A ile B'nin tile'larını shared memory'ye bir kez yükler"><rect x="125.0" y="20" width="70" height="28" rx="4" class="soft-p" stroke="var(--line)"/><text x="160" y="38" text-anchor="middle" class="sm">yazmaçlar</text><rect x="100.0" y="56" width="120" height="28" rx="4" class="soft-p" stroke="var(--line)"/><text x="160" y="74" text-anchor="middle" class="sm">shared memory / L1</text><rect x="75.0" y="92" width="170" height="28" rx="4" class="soft-a" stroke="var(--line)"/><text x="160" y="110" text-anchor="middle" class="sm">L2</text><rect x="50.0" y="128" width="220" height="28" rx="4" class="soft-a" stroke="var(--line)"/><text x="160" y="146" text-anchor="middle" class="sm">HBM (global)</text><text x="280" y="38" class="sm">SM başına 256 KiB · 1 döngü</text><text x="280" y="74" class="sm">SM başına 192 KiB · ~30 döngü</text><text x="280" y="110" class="sm">40 MiB (tüm GPU) · ~200 döngü</text><text x="280" y="146" class="sm">80 GiB, ≈ 2 TB/s · ~500 döngü</text><text x="10" y="172" class="sm">A100 (yaklaşık; gecikmeler mertebe olarak)</text><text x="470" y="20" class="sm">C = A B, n = 4, T = 2</text><rect x="470" y="30" width="20" height="20" class="soft-b" fill-opacity="0.35"/><rect x="492" y="30" width="20" height="20" class="soft-b" fill-opacity="0.35"/><rect x="514" y="30" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="536" y="30" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="470" y="52" width="20" height="20" class="soft-b" fill-opacity="0.35"/><rect x="492" y="52" width="20" height="20" class="soft-b" fill-opacity="0.35"/><rect x="514" y="52" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="536" y="52" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="470" y="74" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="492" y="74" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="514" y="74" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="536" y="74" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="470" y="96" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="492" y="96" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="514" y="96" width="20" height="20" class="mute-f" fill-opacity="0.15"/><rect x="536" y="96" width="20" height="20" class="mute-f" fill-opacity="0.15"/><text x="470" y="134" class="sm">bir block = bir 2×2 C tile’ı</text><text x="470" y="150" class="sm">k adımı başına A ve B’den</text><text x="470" y="166" class="sm">birer 2×2 tile shared’a</text><text x="470" y="182" class="sm">okuma: 4 × 2 × 8 = 64 (naif 128)</text></svg><figcaption>Şekil 2 — Solda bellek hiyerarşisi (yukarı çıktıkça küçük ve hızlı), sağda döşemenin el hesabı. Shared memory'ye bir kez yüklenen her eleman T kez kullanılır; global okumalar T kat azalır.</figcaption></figure>

:::türetim Döşeme okumaları T kat azaltır
$C$'yi $T\times T$'lik tile'lara bölelim; her block bir tile hesaplasın. Block, $k$ boyunca $n/T$ adımda ilerler. Her adımda $A$'dan ve $B$'den birer $T\times T$ tile'ı shared memory'ye yükler ($2T^2$ okuma), `__syncthreads()` ile bekler, sonra tile'lar üzerinden kısmi toplamları hesaplar. Toplam okuma
$$\Big(\frac nT\Big)^2\text{ block}\times\frac nT\text{ adım}\times2T^2=\frac{2n^3}T .$$
Her yüklenen eleman shared memory'den $T$ kez okunur. Adım başına $2T^3$ FLOP yapılır ve $2T^2e$ bayt taşınır: tile'ın aritmetik yoğunluğu $T/e$ FLOP/bayt'tır.
:::

**El hesabı.** $4\times4$ çarpım: naif $16\cdot2\cdot4=\mathbf{128}$ okuma; $2\times2$ tile ile 4 block × 2 adım × (4 + 4) = $\mathbf{64}$.

**Sınırlar.** Thread başına bir çıktıyla $T^2\le1024$, yani $T\le32$. Shared memory de sınırlıdır (A100'de SM başına 192 KiB; block başına varsayılan 48 KiB, isteğe bağlı 164 KiB). fp32'de $T=32$ ile tile yoğunluğu $32/4=8$ FLOP/bayt'tır. Bu, A100'ün fp32 CUDA çekirdekleri için dirsek noktasının (≈ 19.5 TFLOP/s / 2 TB/s ≈ 10) altındadır: shared memory döşemesi tek başına hesap bağımlı bölgeye ulaşamaz. Hızlı çekirdekler bu yüzden **yazmaç döşemesi** de yapar: her thread çıktının küçük bir bloğunu (ör. 8 × 8) yazmaçlarda biriktirir ve etkin tile 128 × 128'e çıkar. Simon Boehm'ün yazısı naif çekirdekten cuBLAS'ın %90'ına bu adımlarla ulaşır. Tensör çekirdekleri (bf16/fp16/tf32) ise ayrı, çok daha yüksek bir tavan getirir.

::lab l2

### Roofline'dan bakış {#roofline}

M6.5'teki roofline modeli burada da geçerlidir. $n=1024$ fp32 matris çarpımının ideal aritmetik yoğunluğu, üç matris birer kez taşınırsa $\frac{2n^3}{3n^2\cdot4}=\frac{2048}{12}=\mathbf{170.67}$ FLOP/bayt'tır: dirseğin çok üstünde, hesap bağımlı. Ama bu **ideal** değer ancak döşeme yeterince iyiyse elde edilir. Naif çekirdeğin gerçek yoğunluğu $1/e$'dir (eleman başına 2 FLOP, 2 okuma). Döşeme, bir çekirdeğin roofline üzerindeki noktasını sağa taşımanın yoludur.

### C++ ve CUDA uzantısı {#uzanti}

PyTorch'a kendi operatörünü eklemenin en kısa yolu `torch.utils.cpp_extension.load_inline`'dır. C++ (ve isteğe bağlı CUDA) kaynak metnini alır, derler ve Python modülü olarak yükler.

```python
from torch.utils.cpp_extension import load_inline
src = r'''
torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor b) {
  TORCH_CHECK(x.dim() == 2 && b.dim() == 1 && x.size(1) == b.size(0), "boyut uyuşmuyor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat && x.is_contiguous(), "fp32, bitişik olmalı");
  auto y = torch::empty_like(x);
  const float* px = x.data_ptr<float>(); ...   // ham işaretçilerle iki döngü
  return y;
}'''
mod = load_inline(name="bias_gelu_ext", cpp_sources=src, functions=["bias_gelu"])
```

- **Doğrulama**: `TORCH_CHECK(koşul, mesaj)` Python'da `RuntimeError` olarak görünür. Ham işaretçilerle çalışırken boyut, dtype, cihaz ve **bitişiklik** mutlaka denetlenir; bitişik olmayan bir görünümün `data_ptr`'ı yanlış elemanları okur.
- **CUDA**: `cuda_sources=` ile bir `__global__` çekirdek ve onu `<<<blocks, threads>>>` ile çağıran bir C++ sarmalayıcı verilir. Çekirdek eşzamansız başlar; hatalar ancak sonraki bir eşzamanlamada görünür (`cudaGetLastError`, `CUDA_LAUNCH_BLOCKING=1`).
- **Autograd**: uzantı yalnız ileri işlemi tanımlar. Türevi için ya bir `autograd.Function` (M6.2) sarmalayıcısı ya da `torch.library` ile kayıtlı bir geri fonksiyon gerekir.
- **torch.compile ile uyum**: `torch.library.custom_op` ile kaydedilen operatörler, bir "sahte" (fake) gerçekleştirimle şekil bilgisini Dynamo'ya verir ve grafı bölmez. Düz bir `load_inline` fonksiyonu graph break'e yol açar.
- **Derleme süresi**: ilk derleme 10–60 saniye sürer, sonra sonuç önbellekten gelir (`~/.cache/torch_extensions`).

Bu yol, bir işlemin NumPy/PyTorch karşılığı yoksa ya da füzyonlu bir çekirdek elle yazılmak isteniyorsa kullanılır. Çoğu durumda Triton (M7.2) aynı işi Python'dan, daha az kodla yapar.

### CPU'da da aynı ilke: önbellek {#cpu}

Döşemenin arkasındaki ilke, yani veriyi hızlı bellekte tutup tekrar kullanmak, CPU'da da geçerlidir. Satır-ana $C=AB$ için üçlü döngünün sırası önemlidir.
- **ijk** (en içte $k$): $B[k][j]$'yi $k$ boyunca okur, yani adım $n$ ile; her erişim ayrı bir önbellek satırına (64 bayt) düşer.
- **ikj** (en içte $j$): $A[i][k]$ sabit, $B[k][\cdot]$ ve $C[i][\cdot]$ satırları ardışık okunur ve yazılır; derleyici vektörleştirebilir.
- **Döşenmiş ikj**: $i$, $k$, $j$ blokları önbelleğe sığacak biçimde seçilir.

Aynı FLOP sayısıyla bu üç sürüm arasında büyük süre farkları ölçülür. BLAS kütüphaneleri (MKL, OpenBLAS) çok seviyeli döşeme ve SIMD ile bunun da ötesine geçer. Aşama e bunu C++'ta ölçüyor.

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| C++ operatörü | `load_inline(name, cpp_sources=src, functions=["f"])` |
| CUDA çekirdeği | `load_inline(..., cuda_sources=cu, functions=[...])`; çekirdekte `__global__`, çağrıda `<<<(n + 255) / 256, 256>>>` |
| girdi denetimi | `TORCH_CHECK(x.is_contiguous(), "...")`, `x.scalar_type() == torch::kFloat` |
| ham işaretçi | `x.data_ptr<float>()`; güvenli erişim: `x.accessor<float, 2>()` |
| derleyiciyle uyum | `@torch.library.custom_op("ns::op", mutates_args=())` + `@op.register_fake` |
| eşzamanlama | `torch.cuda.synchronize()`; hata ayıklama: `CUDA_LAUNCH_BLOCKING=1` |
| profil | `torch.profiler.profile(activities=[ProfilerActivity.CUDA])`, Nsight Compute |

### Özet {#ozet}

- GPU: çok sayıda SM, 32'lik warp'lar (SIMT), gecikmeyi doluluğun gizlemesi; verim odaklı.
- grid → block (≤ 1024 thread, aynı SM, shared memory) → thread; $i=\texttt{blockIdx}\cdot\texttt{blockDim}+\texttt{threadIdx}$, sınır kontrolü. El hesabı: 4 block, 24 boşta, $i=517$.
- Birleşik erişim: ardışık thread'ler ardışık adreslere; adımlı erişimde verim $\approx1/s$, en kötü $e/32$.
- Döşeme: global okuma $2n^3\to2n^3/T$ (4 × 4'te 128 → 64); tile yoğunluğu $T/e$. Thread başına bir çıktıyla $T\le32$ fp32'de dirseğe yetmez; yazmaç döşemesi gerekir.
- İdeal fp32 matmul yoğunluğu $n=1024$'te 170.67 FLOP/bayt; ona ancak iyi döşemeyle yaklaşılır.
- `load_inline` ile C++/CUDA operatörü; `TORCH_CHECK`, bitişiklik, autograd için Function, derleyici için `torch.library.custom_op`.
- CPU'da döngü sırası (ikj) ve döşeme aynı ilkedir.
