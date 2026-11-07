NumPy, **döngü yok** — yalnız iterasyon döngüsü (`for`/`while` bir tane) serbest.

- `kmeans_np(Z, init, max_iter=300)` — `(N, d)` `float64`; `sklearn.cluster.KMeans(init=init, n_init=1, tol=0, algorithm="lloyd")` ile aynı etiketler ve merkezler. İpucu: uzaklıklar `((Z[:, None] - m) ** 2).sum(-1)`, yeni ortalamalar `np.add.at` ya da `np.bincount(labels, weights=…)` ile.
- `quantize(img, k, seed=0)` — RGB `uint8`: `rng = np.random.default_rng(seed)`, başlangıç `Z[rng.choice(N, k, replace=False)]`; `kmeans_np`; her pikseli merkez rengine boya (`np.rint`, `uint8`). `(out, labels (H, W))`.
