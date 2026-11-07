Doğrusal autoencoder: $Z=XW_e$, $\hat X=ZW_d$, $L=\frac1n\lVert\hat X-X\rVert_F^2$. Gradyanları kendin yaz:

$$
G=\tfrac2n(\hat X-X),\quad \nabla_{W_d}=Z^\top G,\quad \nabla_{W_e}=X^\top G\,W_d^\top.
$$

- `ae_grads(X, We, Wd)` → `(L, dWe, dWd)`. Döngü yok.
- `train_linear_ae(X, k, steps=3000, lr=0.02, seed=0)` → `(We, Wd, L)`; başlangıç `0.1 * normal`. Tek döngü (adımlar).
- `pca_error(X, k)` — en iyi rank-$k$ yeniden yapılandırmanın hatası: $X^\top X/n$'nin en küçük $d-k$ özdeğerinin toplamı.

Kontrol, eğitilen autoencoder'ın PCA optimumuna %1 içinde indiğini sınar.
