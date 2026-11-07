Yerel minimum riskine karşı standart çözüm: $R$ farklı başlatmayla koş, en düşük inertia'yı seç. $R$'yi bir batch boyutu yap:

- merkezler `(R, k, d)`, mesafeler `(R, n, k)`, atamalar `(R, n)`,
- one-hot `(R, n, k)`, toplamlar `torch.einsum("rnk,nd->rkd", A, X)`, sayımlar `A.sum(1)` `(R, k)`.

- `kmeans_restarts(X, C0s, iters)` → `(best_C: (k, d), inertias: (R,))`. Yalnız iterasyon döngüsü; başlatmalar üzerinde döngü yok. Boş küme eski merkezini korur.

Kontrol, her başlatmayı `kmeans_torch` ile tek tek koşturup karşılaştırır.
