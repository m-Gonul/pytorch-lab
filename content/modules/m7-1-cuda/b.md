Saf Python, CUDA'yı taklit eden bir simülatör. `Ctx` (block/thread indeksleri, `blockDim`) ve okuma sayan `Global` hazır.

- `launch(kernel, grid, block, *args)` — 1-boyutlu: her `b in range(grid)`, `t in range(block)` için `kernel(Ctx(b, t, block), *args)`.
- `vec_add(ctx, x, y, out, n)` — thread başına bir eleman, **sınır kontrolüyle**.
- `matmul_naive(A, B, n)` → `C` (düz liste, satır-ana); `A`, `B` `Global`.
- `matmul_tiled(A, B, n, T)` — her $(b_i, b_j)$ block'u için k-tile'larını yerel ("shared") listelere **bir kez** yükle, sonra o tile'lardan hesapla. $n$, $T$'ye bölünür.

Kontrol okuma sayılarını $2n^3$ ve $2n^3/T$ ile karşılaştırır.
