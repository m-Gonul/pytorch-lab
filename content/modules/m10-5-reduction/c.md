NumPy; döngü yalnız ağaç seviyeleri üzerinde.

- `tree_reduce(x, op, identity)` — `op` bir ufunc (`np.add`, `np.maximum`, …); her seviyede `op(x[0::2], x[1::2])`, tek uzunlukta birim ile doldur.
- `naive_sum32(x)` (float32 sıralı: `np.cumsum(dtype=float32)[-1]`) ve `tree_sum32(x)` (float32 ağaç).
- `argmax_tree(x)` — (değer, indeks) çiftlerinin ağaç indirgemesi; eşitlikte küçük indeks.
- `chain_product_tree(M)` — `M (n, 2, 2)` matrisler, sırayı koruyan ağaç çarpımı (`np.einsum`/`@`).
