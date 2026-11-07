- `pca(X, k)` → `(components: (k, d), variances: (k,), Z: (n, k))`. Ortala, `np.linalg.svd(Xc, full_matrices=False)`, ilk $k$ satır `Vt`, varyans $s^2/(n-1)$, işaret kuralını vektörel uygula (`np.argmax(np.abs(V), axis=1)` ile her satırın en büyük elemanı). $Z=X_cV^\top$.
- `reconstruct(Z, components, mean)` → `(n, d)`.

**Döngü yok.** Kontrol: bileşenler ortonormal, varyanslar kovaryansın özdeğerleriyle aynı, tam $k=d$'de yeniden kurma kayıpsız.
