CS231n arayüzü, batch'li, satır vektörleri: `x (N, T, D)`, `h0 (N, H)`, `Wx (D, H)`, `Wh (H, H)`, `b (H,)`.

- `rnn_forward(x, h0, Wx, Wh, b)` → `(h, cache)`, `h: (N, T, H)`.
- `rnn_backward(dh, cache)` → `(dx, dh0, dWx, dWh, db)`; `dh: (N, T, H)` her adımın doğrudan gradyanı.

Yalnız zaman döngüsü (adım içinde döngü yok).
