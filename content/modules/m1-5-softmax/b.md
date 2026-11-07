Saf Python, tek örnek. `math` serbest; `numpy`/`torch` yasak.

- `logsumexp(z)` — kararlı (max çıkararak).
- `softmax(z)` — kararlı; `logsumexp` üzerinden yazabilirsin: $p_k=e^{z_k-\text{LSE}(z)}$.
- `cross_entropy(z, y)`
- `grad_z(z, y)` — $p-e_y$ listesi.

Kontrol $z=[1000,999,998]$ ve $z=[-1000,-1001]$ ile taşma olmadığını, gradyanın sayısal türevle eşleştiğini doğrular.
