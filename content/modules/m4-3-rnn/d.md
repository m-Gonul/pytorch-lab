- `MyRNN(input_size, hidden_size)` — `nn.Module`, `batch_first=True` davranışı; parametre adları `nn.RNN` ile aynı: `weight_ih_l0 (H, D)`, `weight_hh_l0 (H, H)`, `bias_ih_l0`, `bias_hh_l0`. `forward(x, h0=None)` → `(out: (N, T, H), h_n: (1, N, H))`. Böylece `my.load_state_dict(torch_rnn.state_dict())` çalışır.
- `input_grad_norms(rnn, x)` → uzunluğu $T$ liste: $\lVert\partial\,(\sum h_T)/\partial x_t\rVert$, $t=1..T$ (her örnek ve özellik üzerinden norm).

Kontrol `nn.RNN` ile çıktı ve gradyan eşitliğini, sonra 50 adımlık bir dizide gradyanın geçmişe doğru sönmesini gösterir.
