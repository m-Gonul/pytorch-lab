Saf Python; vektörler liste, matrisler satır listesi. **Matris düzeni:** $W_x$ $(H\times D)$, $W_h$ $(H\times H)$, $h_t=\tanh(W_xx_t+W_hh_{t-1}+b)$.

- `rnn_forward(xs, h0, Wx, Wh, b)` → `hs` (uzunluk $T$ liste; $h_1..h_T$).
- `rnn_backward(xs, hs, h0, Wx, Wh, dhs)` — `dhs[t]`: kaybın $h_{t+1}$'e **doğrudan** gradyanı. → `(dWx, dWh, db, dh0)`. Zamanda geriye yürü, dolaylı gradyanı her adımda ekle.
