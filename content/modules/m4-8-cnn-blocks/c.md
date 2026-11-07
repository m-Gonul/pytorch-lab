NumPy, döngüsüz. Görüntüler `(C, H, W)`, "same" dolgu, adım 1.

- `conv2d(x, w)` — `w (Cout, Cin, K, K)` (`sliding_window_view` + `einsum`).
- `depthwise(x, w)` — `w (C, K, K)`; `pointwise(x, w)` — `w (Cout, Cin)`.
- `grouped_conv(x, w, g)` — `w (Cout, Cin/g, K, K)`.
- `se_block(x, W1, b1, W2, b2)` — GAP → ReLU(W1 · + b1) → sigmoid(W2 · + b2) → kanal ölçekleme.
- `lrn(x, k=2, n=5, alpha=1e-4, beta=0.75)` — kanal ekseninde kayan kareler toplamı (kümülatif toplamla).
