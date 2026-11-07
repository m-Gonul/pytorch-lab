Saf Python, tek örnek; matrisler satır listesi. PyTorch düzeni: `W_ih (4H × D)`, `W_hh (4H × H)`, `b_ih`, `b_hh (4H,)`; satır blokları LSTM'de `i, f, g, o`, GRU'da `(3H)` ve `r, z, n`.

- `lstm_step(x, h, c, W_ih, W_hh, b_ih, b_hh)` → `(h_yeni, c_yeni)`
- `gru_step(x, h, W_ih, W_hh, b_ih, b_hh)` → `h_yeni`. $n$'de $r$, yalnız $W_{hn}h+b_{hn}$ kısmını çarpar.
