NumPy, **döngü yok**:

- `embed_backward(ids, dout, V)` → `dE: (V, d)`; `np.add.at(dE, ids, dout)` (aynı id'lerin katkılarını toplar; `dE[ids] += dout` toplamaz).
- `pair_counts_np(ids, V)` → `(V, V)` sayım matrisi; çiftleri tek tamsayıya kodla: `ids[:-1] * V + ids[1:]`, sonra `np.bincount(…, minlength=V*V)`.
- `merge_ids(ids, a, b, new)` — `a ≠ b` için `(a, b)` geçişlerini `new` ile değiştir (bu durumda eşleşmeler örtüşemez). Eşleşme maskesi `m = (ids[:-1] == a) & (ids[1:] == b)`; eşleşmenin ilk elemanını `new` yap, ikincisini sil.
