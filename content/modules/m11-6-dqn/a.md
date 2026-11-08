<!-- çözüm -->

Doğrusal: $\hat q=0.5+1=1.5$, $e=3-1.5=1.5$, $w\leftarrow w+0.1\cdot1.5\cdot x=(0.65,0.8)$.

Huber: $|e|\le1$ için $\tfrac12e^2$, yoksa $|e|-\tfrac12$ → türev $\mathrm{clip}(e,-1,1)$. DQN'nin "hata kırpma"sı budur.

Conv çıktısı: $\lfloor(n-k)/s\rfloor+1$: 84 → 20 → 9 → 7, $7\cdot7\cdot64=3136$ (sonra 512 tam bağlı, sonra eylem başına bir çıktı — tek ileri geçişte tüm Q'lar).
