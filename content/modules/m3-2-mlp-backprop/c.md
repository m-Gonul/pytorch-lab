CS231n modüler katmanları NumPy'da, **batch**; tüm katman fonksiyonlarında **döngü yok**:

- `affine_forward(x, w, b)` → `(out, cache)`; `affine_backward(dout, cache)` → `(dx, dw, db)`
- `relu_forward(x)`, `relu_backward(dout, cache)`
- `softmax_loss(scores, y)` → `(loss, dscores)`
- `two_layer_loss(X, y, params, reg=0.0)` → `(loss, grads)`; `params`: `W1 (D, H)`, `b1`, `W2 (H, C)`, `b2`; ceza `reg·(‖W1‖² + ‖W2‖²)`.
- `train(X, y, H, C, lr=1.0, steps=5000, reg=1e-3, seed=0)` → `params`; başlatma `0.01 * rng.standard_normal`, tam batch, yalnız adım döngüsü.

Kontrol, CS231n'in spiral verisinde (3 kol) eğitim doğruluğunun ≥ 0.95 olduğunu doğrular. Lineer softmax bu veride ~%50'de kalır.
