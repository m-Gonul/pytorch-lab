NumPy, birleşik softmax + cross-entropy. **Döngü yok.**

- `ce_forward(logits, y)` → `(kayıp, cache)`; `logits: (N, C)`, `y: (N,)`. Kararlı log-softmax. `cache` yalnız olasılıkları ve `y`'yi tutsun.
- `ce_backward(cache)` → `dlogits = (p - onehot(y)) / N`.

Bir de `dir_gradcheck(f, grad, x, rng, k=10, eps=1e-6)` — rastgele $k$ yön $v$ için $\frac{f(x+\epsilon v)-f(x-\epsilon v)}{2\epsilon}$ ile $\langle\nabla f, v\rangle$'yi karşılaştırır, en büyük göreli hatayı döndürür. (Yönler üzerinde tek döngü serbest.) Tam gradcheck'in $O(d)$ maliyeti yerine $O(k)$.
