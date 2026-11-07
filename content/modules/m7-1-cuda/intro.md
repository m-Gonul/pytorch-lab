**CUDA yürütme modeli.** Bir çekirdek (kernel) binlerce thread'de aynı anda çalışır. Thread'ler **block**'lara, block'lar **grid**'e gruplanır:

$$
i=\texttt{blockIdx.x}\cdot\texttt{blockDim.x}+\texttt{threadIdx.x}.
$$

Aynı block'taki thread'ler hızlı **shared memory**'yi paylaşır ve `__syncthreads()` ile eşzamanlanır; global bellek (HBM) büyük ama yavaştır.

**[[tiling]].** Naif matmul'de her $C_{ij}$, $A$'nın bir satırını ve $B$'nin bir sütununu global bellekten okur: toplam $2n^3$ okuma. $T\times T$ tile'larla bir block, $A$ ve $B$'nin tile'larını bir kez shared memory'ye yükler ve $T$ kez yeniden kullanır: $2n^3/T$.

**[[roofline]].** Aritmetik yoğunluk $I=\text{FLOP}/\text{bayt}$. Ulaşılabilir performans $\min(\text{tepe FLOP},\ \text{bant genişliği}\times I)$. $I$ tepe noktasının altındaysa işlem **bellek-bağımlı**dır.

**Extension.** `torch.utils.cpp_extension.load_inline` bir C++ (ve isteğe bağlı CUDA) kaynağını derleyip Python modülü olarak yükler; fonksiyonlar `torch::Tensor` alır ve döndürür.

Bu modül GPU olmadan da çalışır: simülasyon (b), model (c) ve C++ (d, e) CPU'da. GPU varsa d aşamasında aynı fonksiyonun CUDA hâli de denenir.
