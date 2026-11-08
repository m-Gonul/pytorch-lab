NumPy. Bağlı ağırlıklı, sigmoid kodlu, doğrusal çıkışlı seyrek AE: $z=\sigma(XW+b)$, $\hat X=zW^\top+c$,
$$L=\frac1{2N}\|\hat X-X\|^2+\beta\sum_j\mathrm{KL}(\rho\Vert\hat\rho_j),\quad\hat\rho_j=\tfrac1N\sum_iz_{ij}.$$

- `tied_sparse_ae(X, W, b, c, rho, beta)` → `(L, dW, db, dc)` — döngüsüz; $W$ hem kodlayıcıda hem kod çözücüde: iki gradyan katkısı.
- `train(X, k, rho, beta, lr=0.5, iters=1500, seed=0)` → `(W, b, c)` — tam batch gradyan inişi (tek döngü).
