NumPy; doğrusal regresyon $y=Xw^*$, tam batch GD. **Yalnız adım döngüsü.**

- `train_fp16(X, y, lr, steps, master)` → son kayıp (fp32'de hesaplanmış). Ağırlıklar `np.float16`; ileri geçiş ve gradyan fp16'da.
  - `master=False`: `w = (w - lr * g).astype(np.float16)`.
  - `master=True`: fp32 bir `w32` tut, `w32 -= lr * g.astype(np.float32)`, ileri geçiş için `w = w32.astype(np.float16)`.
- `scaled_grad(g, scale)` → `(np.float16(g), np.float16(g * scale) / scale)` (ikincisi fp32'de bölünür): alt taşmanın ölçekle önlenmesi.
