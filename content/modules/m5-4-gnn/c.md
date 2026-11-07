NumPy; `edge_index: (2, E)` iki yönlü, öz-döngüsüz. **Döngü yok.**

- `dense_norm_adj(edge_index, n)` → $\hat A$ (yoğun): `A[src, dst] = 1`, `+ I`, derece, normalize.
- `propagate(edge_index, X, n)` → $\hat AX$ yoğun matris kurmadan: öz-döngüleri ekle, dereceyi `np.bincount` ile al, kenar ağırlıkları $1/\sqrt{\tilde d_s\tilde d_t}$, `np.add.at(out, dst, w[:, None] * X[src])`.

Kontrol ikisinin aynı olduğunu ve permütasyon eşdeğişkenliğini sınar.
