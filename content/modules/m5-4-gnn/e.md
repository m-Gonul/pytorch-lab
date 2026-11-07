Graf sınıflandırma için PyG tarzı batching. `propagate` hazır verildi.

- `collate(graphs)` — `graphs`: `(x_i, edge_index_i)` listesi → `(X, EI, batch)`. Düğümleri art arda ekle, her grafın kenarlarına o grafın **düğüm ofsetini** ekle (`cumsum`), `batch[k]` = düğüm $k$'nin grafı (`repeat_interleave`). Graflar üzerinde tek seviye döngü/comprehension serbest.
- `readout(x, batch, B)` → `(B, 2d)`: graf başına `[ortalama, max]`. **Döngü yok:** `index_add_` ve `scatter_reduce` (ya da `index_reduce_`, `"amax"`).

Kontrol, birleşik grafla hesaplanan graf temsillerinin grafları tek tek işlemekle birebir aynı olduğunu doğrular.
