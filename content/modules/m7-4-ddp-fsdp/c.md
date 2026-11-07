NumPy simülasyonu, doğrusal regresyon $\ell=\frac1n\lVert Xw-y\rVert^2$.

- `worker_grads(w, X, y, K)` → $K$ işçinin kendi (eşit) parçalarındaki gradyan listesi.
- `allreduce_mean(grads)` → ortalama.
- `adam_update(w, g, m, v, t, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8)` → `(w, m, v)`; bias düzeltmeli. Döngü yok.
- `zero1_step(w, g, states, t, lr=1e-2)` — `states`: işçi başına `(m_k, v_k)`, her biri $w$'nin $k$. parçası boyunda (`np.array_split`). Her işçi kendi parçasını günceller, sonra parçalar birleştirilir (all-gather). → `(w, states)`.
