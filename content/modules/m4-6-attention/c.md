NumPy, **döngü yok**. `x: (B, T, d)`, ağırlıklar `Wq, Wk, Wv, Wo: (d, d)` (çarpım `x @ W`).

- `softmax(x, axis=-1)` — kararlı, `-inf` destekli.
- `mha(x, Wq, Wk, Wv, Wo, h, causal=False)` → `(B, T, d)`.

Yol: `q = x @ Wq` → `(B, T, h, dk)` → `transpose(0, 2, 1, 3)` → `(B, h, T, dk)`; skorlar `q @ k.transpose(0, 1, 3, 2)`; maske `np.triu(…, 1)`; sonuç geri `(B, T, d)` ve `@ Wo`.
