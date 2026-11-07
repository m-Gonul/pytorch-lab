NumPy, batch'li, satır vektörleri: `Wx (D, 4H)`, `Wh (H, 4H)`, `b (4H,)`, sütun blokları `i, f, g, o`. **Döngü yok.**

- `lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)` → `(next_h, next_c, cache)`
- `lstm_step_backward(dnext_h, dnext_c, cache)` → `(dx, dprev_h, dprev_c, dWx, dWh, db)`

Geri geçişte dört kapının ön-aktivasyon gradyanlarını `(N, 4H)` bir matrise yan yana koy; sonra tek seferde $dx=da\,W_x^\top$, $dW_x=x^\top da$.
