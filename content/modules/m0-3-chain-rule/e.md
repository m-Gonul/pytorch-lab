Aynı türevi $N$ farklı noktada istiyoruz (ör. bir grafikte gradyan alanı çizmek, ya da örnek başına türev). Döngüyle $N$ kez backward çağırmak yerine iki yol:

1. **Toplam hilesi:** noktalar birbirinden bağımsızsa $\sum_k f(p_k)$'nin $p_k$'ya göre gradyanı tam $\nabla f(p_k)$'dır. Tek `backward`.
2. **`torch.func`:** tek nokta için yazılmış dönüşümü [[vmap]] ile batch'e genişlet:

```python
from torch.func import grad, jacrev, vmap
vmap(grad(f_scalar))(P)    # (N, 2)
vmap(jacrev(g_vec))(P)     # (N, 2, 2)
```

- `batch_grad(P)` — `P: (N, 2)` → her satırda $(\partial f/\partial x,\partial f/\partial y)$.
- `batch_jacobian(P)` — `(N, 2) → (N, 2, 2)`.

İkisinde de **döngü yok**. Kontrol, 512 nokta için `for` + `jacobian` ile `vmap(jacrev)`'i yarıştırır.
