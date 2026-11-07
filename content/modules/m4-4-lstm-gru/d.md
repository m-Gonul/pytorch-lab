- `MyLSTMCell(input_size, hidden_size)` — parametre adları `nn.LSTMCell` ile aynı: `weight_ih (4H, D)`, `weight_hh (4H, H)`, `bias_ih`, `bias_hh`. `forward(x, state=None)` → `(h, c)`; `state` yoksa sıfırlar.
- `run_cell(cell, x)` — `x: (N, T, D)`; hücreyi zamanda çalıştır → `(out: (N, T, H), (h_T, c_T))`.

Kontrol `nn.LSTMCell` ile tek adım eşitliğini, `nn.LSTM` (state_dict'i `_l0` adlarıyla kopyalayarak) ile tüm dizi eşitliğini ve gradyanları denetler.
