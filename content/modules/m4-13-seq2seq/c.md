NumPy, döngüsüz.

- `masked_softmax(scores, mask)` — `mask` False olan konumlar 0 ağırlık.
- `additive_attention(s, H, W1, W2, v, mask)` — `s (B, d_s)`, `H (B, T, d_h)`, `W1 (d_h, d_a)`, `W2 (d_s, d_a)`, `v (d_a,)`: $e_{t,i}=v^\top\tanh(W_1^\top h_i+W_2^\top s)$ → `(α (B, T), bağlam (B, d_h))`.
- `luong_attention(s, H, mask, W=None)` — `W=None`: nokta; değilse "general" $s^\top W h_i$.
- `masked_ce(logits, targets, pad=0)` — `(B, T, V)`; dolgu hedefleri hariç ortalama çapraz entropi.
