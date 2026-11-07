`iwae_bound(encode, decode, x, eps)` — `eps: (K, B, L)`. Hepsi tek seferde:

1. `mu, logvar = encode(x)`; `z = mu + std * eps` → `(K, B, L)` (yayınlama).
2. `decode(z.reshape(K * B, L))` → `(K, B, d)`.
3. $\log w_k=\log p(x\mid z_k)+\log p(z_k)-\log q(z_k\mid x)$; $p(x\mid z)=\mathcal N(\hat x, I)$.
4. Sonuç: $\operatorname{mean}_b\big[\operatorname{logsumexp}_k\log w_k-\log K\big]$.

**Döngü yok** ($K$ üzerinde de). Kontrol, $K=1$'de ELBO'ya eşitliği, döngülü başvuruyla aynılığı ve sınırın $K$ ile arttığını sınar.
