Lloyd güncellemesi neden ortalama? Atamalar sabitken $J$, $c_k$'ye göre bir kuadratik:

$$
\frac{\partial J}{\partial c_k}=2\sum_{i:z_i=k}(c_k-x_i)=2\big(n_kc_k-S_k\big)=0\ \Rightarrow\ c_k=\frac{S_k}{n_k}.
$$

Yani güncelleme, gradyanı sıfırlayan tam bir **Newton adımı**: $c_k\leftarrow c_k-\frac{1}{2n_k}\frac{\partial J}{\partial c_k}$.

- `kmeans_torch(X, C0, iters)` → `(C, labels, inertia)`; atama `torch.cdist(X, C).argmin(1)`, toplamlar `torch.zeros_like(C).index_add_(0, labels, X)`, sayımlar `torch.bincount(labels, minlength=k)`.
- `lloyd_step_via_grad(X, C)` → yeni merkezler; atamaları hesapla, $J$'yi **autograd** ile türet, Newton adımını uygula (boş kümede değişiklik yok). Kontrol, sonucun ortalama güncellemesiyle aynı olduğunu doğrular.
