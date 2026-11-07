Gerçek iş yükünde tek bir matris değil, bir **batch** (yığın) matris çarpılır: $N$ adet $(n\times k)$ ile $N$ adet $(k\times m)$. Python'da `for` ile $N$ kez `@` çağırmak her adımda yorumlayıcı ve kernel başlatma maliyeti öder.

`batched_matmul(As, Bs)` yaz: `As.shape == (N, n, k)`, `Bs.shape == (N, k, m)`, sonuç `(N, n, m)`. **Döngü yasak.** Üç eşdeğer yol:

```python
As @ Bs                              # matmul baştaki boyutları batch sayar
torch.bmm(As, Bs)                    # yalnız 3-boyutlu, açık batch
torch.einsum("bik,bkj->bij", As, Bs)  # indis notasyonu: tanımın kendisi
```

`einsum` dizgesi (b) aşamasındaki döngünün harfiyen tarifidir: tekrar eden ama çıktıda olmayan `k` toplanır.

Kontrol ayrıca bir **hız tablosu** üretir: aynı $64\times64$ çarpımı saf Python döngüsü, NumPy ve PyTorch ile; ve 256 küçük matris için döngü vs batched. Sonuçları yorumla: hızlanmanın ne kadarı yorumlayıcı maliyetinin kalkmasından, ne kadarı [[BLAS]] çekirdeklerinden geliyor?
