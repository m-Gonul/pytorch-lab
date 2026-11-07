CS231n'in `softmax_loss_vectorized`'ı: $n$ örnek, $K$ sınıf, **döngü yok**.

- `softmax_rows(Z)` — `(n, K)` satır satır kararlı softmax (`keepdims=True` ile max çıkar).
- `softmax_loss(W, X, y, reg=0.0)` → `(loss, dW)`; `W: (d, K)`, bias yok (X'in son sütunu 1 olabilir). Kayıp: ortalama cross-entropy $+\ \text{reg}\cdot\lVert W\rVert^2$. Doğru sınıfın log-olasılığını almak için gelişmiş indeksleme: `logP[np.arange(n), y]`. One-hot için `np.eye(K)[y]` ya da `P[np.arange(n), y] -= 1`.
- `fit_softmax(X, y, K, lr=0.5, steps=300)` → `W`; yalnız adım döngüsü.
