Saf Python, tek özellikli regresyon ($\hat y=wx+b$, MSE). `random` serbest, `numpy`/`torch` yasak.

- `minibatches(n, batch_size, rng)` — `range(n)`'i `rng.shuffle` ile karıştır, `batch_size`'lık parçalara böl, **liste listesi** döndür. Son parça küçük olabilir.
- `sgd_epoch(xs, ys, w, b, lr, batch_size, rng)` → `(w, b)`; her mini-batch'in **ortalama** gradyanıyla bir adım.
- `momentum_path(grad, w0, lr, beta, steps)` → `[w_1, …, w_steps]`; tek değişkenli, `grad` bir fonksiyon.

Kontrol: `batch_size = n` ile tek GD adımı, `batch_size = 1` ile kapalı forma yakınsama, momentum yörüngesi elle bulduğunla aynı.
