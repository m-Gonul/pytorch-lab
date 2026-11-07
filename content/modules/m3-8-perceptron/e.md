**Toplu eğitim:** E adet aynı mimarili MLP'yi tek optimizasyonda eğit — ağırlıklar `(E, d_in, h)` tensörlerinde, ileri geçiş `torch.baddbmm`. Her model kendi verisini (farklı tohum) görür.

- `BatchedMLP(E, d_in, h, depth, d_out)` — `forward(X)`, `X (E, n, d_in)` → `(E, n, d_out)`.
- `train_batched(E, fn, depth, steps=1500, n=1000)` → `(E,)` test MSE'leri.
- `train_loop(E, fn, depth, steps=1500, n=1000)` — aynısını E kez ayrı `nn.Sequential` ile (karşılaştırma için).
